//! Per-run resource limits: a wall-clock deadline and a cap on the bytes a program builds.
//!
//! VRL has no loops, but `.a = .a + .a` doubles a value per statement and `decode_gzip` can
//! inflate a few bytes into gigabytes, so an untrusted program needs a ceiling on both time and
//! memory. The interpreter cooperates: [`Context::tick`] runs on every expression node and
//! [`Context::charge`] at every site that builds a value, so an exceeded limit is a
//! [`ExpressionError::Error`] like any other runtime failure. Because `??` and `, err =` swallow
//! errors, the limit is also sticky: once exceeded every further tick fails again, and the
//! runner reads [`Context::exceeded`] after the run so the program cannot hide the outcome.
//!
//! With no limits set (the default) a run pays one counter decrement per node and no clock reads.
//!
//! [`Context::tick`]: super::Context::tick
//! [`Context::charge`]: super::Context::charge
//! [`Context::exceeded`]: super::Context::exceeded
//! [`ExpressionError::Error`]: super::ExpressionError::Error

use std::time::{Duration, Instant};

use super::ExpressionError;
use crate::value::Value;

/// Which limit a run exceeded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    /// The wall-clock deadline passed.
    Time,
    /// The program built more bytes than allowed.
    Memory,
}

/// The limits a run is subject to. `None` means unlimited.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Limits {
    /// Cap on the cumulative bytes of values the program builds during the run (strings at their
    /// length, containers per node). Cumulative, not live: a value that is built and dropped still
    /// counts, which keeps the accounting to a walk of what was just built. A function that hands
    /// its input back (`string!(.message)`) is charged for it again, so a program that passes a
    /// large field through many calls spends budget it did not allocate; the cap is meant to sit
    /// far above that.
    pub max_bytes: Option<usize>,
    /// Wall-clock deadline, measured from the run's start.
    pub max_duration: Option<Duration>,
}

/// Nodes between two clock reads. A node is tens of nanoseconds, so the deadline is late by at
/// most a few hundred microseconds — and by less than one op after a large allocation, which
/// forces a read at the next tick.
const TICKS_PER_CLOCK_READ: u32 = 1024;

/// A charge, or an input string, at least this big makes the next tick read the clock: expensive
/// nodes are the ones that handle a lot of bytes, so an op that just built or read a lot should
/// not get another 1024 nodes' grace.
const LARGE_CHARGE: usize = 64 * 1024;

/// Estimated bytes of a `Value` node that holds no payload of its own (integer, float, null, …).
pub(crate) const SCALAR_BYTES: usize = 16;
/// Estimated bytes of a `Value::Bytes` handle, excluding the buffer.
const BYTES_HANDLE: usize = 32;
/// Estimated bytes of an empty object, and per entry on top of its key and value.
const OBJECT_BYTES: usize = 48;
const OBJECT_ENTRY_BYTES: usize = 16;
/// Estimated bytes of an empty array.
const ARRAY_BYTES: usize = 32;

/// The running tally for one program run.
#[derive(Debug)]
pub struct Budget {
    limits: Limits,
    ticks_until_clock: u32,
    started: Option<Instant>,
    bytes: usize,
    exceeded: Option<Limit>,
}

impl Budget {
    #[must_use]
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            ticks_until_clock: TICKS_PER_CLOCK_READ,
            // One clock read per limited run, so the deadline means what it says even for a
            // program of a few nodes that each chew through a large input.
            started: limits.max_duration.map(|_| Instant::now()),
            bytes: 0,
            exceeded: None,
        }
    }

    #[must_use]
    pub fn has_memory_limit(&self) -> bool {
        self.limits.max_bytes.is_some()
    }

    /// A node is about to work on `bytes` of input (a large string read from the event or a
    /// variable): make the next tick read the clock rather than wait out the remaining ticks.
    #[inline]
    pub fn note_input(&mut self, bytes: usize) {
        if bytes >= LARGE_CHARGE && self.limits.max_duration.is_some() {
            self.ticks_until_clock = 1;
        }
    }

    /// Count one expression node. Reads the clock every [`TICKS_PER_CLOCK_READ`] nodes, or at the
    /// node after a large charge, and fails once the deadline has passed or a limit was already
    /// exceeded.
    #[inline]
    pub fn tick(&mut self) -> Result<(), ExpressionError> {
        self.ticks_until_clock -= 1;
        if self.ticks_until_clock == 0 {
            return self.check_clock();
        }
        Ok(())
    }

    #[cold]
    fn check_clock(&mut self) -> Result<(), ExpressionError> {
        if let Some(limit) = self.exceeded {
            // Stay failed: a `??` that swallowed the first error gets the next node's error.
            self.ticks_until_clock = 1;
            return Err(error(limit, &self.limits));
        }
        self.ticks_until_clock = TICKS_PER_CLOCK_READ;
        if let (Some(max), Some(started)) = (self.limits.max_duration, self.started) {
            if started.elapsed() > max {
                return Err(self.exceed(Limit::Time));
            }
        }
        Ok(())
    }

    /// Account for `bytes` of newly built value. Fails, and stays failed, past `max_bytes`.
    #[inline]
    pub fn charge(&mut self, bytes: usize) -> Result<(), ExpressionError> {
        self.bytes = self.bytes.saturating_add(bytes);
        if self.exceeded.is_some() || self.bytes > self.limits.max_bytes.unwrap_or(usize::MAX) {
            return Err(self.exceed(Limit::Memory));
        }
        if bytes >= LARGE_CHARGE {
            self.ticks_until_clock = 1;
        }
        Ok(())
    }

    /// Bytes the program may still build before `max_bytes` is exceeded.
    #[must_use]
    pub fn remaining_bytes(&self) -> usize {
        self.limits
            .max_bytes
            .map_or(usize::MAX, |max| max.saturating_sub(self.bytes))
    }

    /// Mark the run as over `limit` (the first one wins) and return its error.
    pub fn exceed(&mut self, limit: Limit) -> ExpressionError {
        let limit = *self.exceeded.get_or_insert(limit);
        self.ticks_until_clock = 1;
        error(limit, &self.limits)
    }

    #[must_use]
    pub fn exceeded(&self) -> Option<Limit> {
        self.exceeded
    }

    #[must_use]
    pub fn limits(&self) -> Limits {
        self.limits
    }

    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Limits {
    /// The message a run that exceeded `limit` fails with. Public so an embedder that reads the
    /// sticky flag after a run (the error itself may have been swallowed) can report the same text.
    #[must_use]
    pub fn message(&self, limit: Limit) -> String {
        match limit {
            Limit::Time => {
                let ms = self.max_duration.unwrap_or_default().as_secs_f64() * 1000.0;
                if ms.fract() == 0.0 {
                    format!("program exceeded the {ms:.0} ms time limit")
                } else {
                    format!("program exceeded the {ms:.1} ms time limit")
                }
            }
            Limit::Memory => {
                let bytes = self.max_bytes.unwrap_or_default();
                let size = if bytes % (1024 * 1024) == 0 {
                    format!("{} MiB", bytes / (1024 * 1024))
                } else if bytes % 1024 == 0 {
                    format!("{} KiB", bytes / 1024)
                } else {
                    format!("{bytes} bytes")
                };
                format!("program exceeded the {size} memory limit")
            }
        }
    }
}

fn error(limit: Limit, limits: &Limits) -> ExpressionError {
    limits.message(limit).into()
}

/// Estimated bytes of a clone of `value`: containers per node, keys at their length, and a
/// `Bytes` at its handle only, because cloning one shares the buffer rather than copying it.
#[must_use]
pub fn clone_size_of(value: &Value) -> usize {
    match value {
        Value::Bytes(_) => BYTES_HANDLE,
        Value::Object(map) => map.iter().fold(OBJECT_BYTES, |total, (key, value)| {
            total + OBJECT_ENTRY_BYTES + key.len() + clone_size_of(value)
        }),
        Value::Array(items) => items
            .iter()
            .fold(ARRAY_BYTES, |total, value| total + clone_size_of(value)),
        Value::Regex(_)
        | Value::Integer(_)
        | Value::Float(_)
        | Value::Boolean(_)
        | Value::Timestamp(_)
        | Value::Null => SCALAR_BYTES,
    }
}

/// Estimated bytes of a value counting only its top level: a string at its length, a container
/// by its element count. For function results, whose depth is bounded by their inputs (every
/// read was charged in full, and the few functions that can inflate bound themselves), so the
/// walk a full estimate would cost on every call is not worth it.
#[must_use]
pub fn shallow_size_of(value: &Value) -> usize {
    match value {
        Value::Bytes(bytes) => BYTES_HANDLE + bytes.len(),
        Value::Object(map) => OBJECT_BYTES + map.len() * (OBJECT_ENTRY_BYTES + SCALAR_BYTES),
        Value::Array(items) => ARRAY_BYTES + items.len() * SCALAR_BYTES,
        Value::Regex(_)
        | Value::Integer(_)
        | Value::Float(_)
        | Value::Boolean(_)
        | Value::Timestamp(_)
        | Value::Null => SCALAR_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{compile, state::RuntimeState, Context, Program, TargetValue, TimeZone};
    use crate::value::Secrets;

    fn compile_program(source: &str) -> Program {
        compile(source, &crate::stdlib::all())
            .unwrap_or_else(|diagnostics| panic!("compile failed: {diagnostics:?}"))
            .program
    }

    /// Runs `source` on `event` under `limits`; returns the limit it exceeded, the final event,
    /// and the program's own result.
    fn run(
        source: &str,
        event: Value,
        limits: Limits,
    ) -> (Option<Limit>, Value, Result<Value, ExpressionError>) {
        let program = compile_program(source);
        let mut target = TargetValue {
            value: event,
            metadata: Value::Object(Default::default()),
            secrets: Secrets::new(),
        };
        let mut state = RuntimeState::default();
        let timezone = TimeZone::default();
        let mut ctx = Context::with_limits(&mut target, &mut state, &timezone, limits);
        let result = program.resolve(&mut ctx);
        (ctx.exceeded(), target.value, result)
    }

    fn bytes(max_bytes: usize) -> Limits {
        Limits {
            max_bytes: Some(max_bytes),
            max_duration: None,
        }
    }

    const MIB: usize = 1024 * 1024;

    #[test]
    fn unlimited_runs_are_untouched() {
        let doubling = ".a = \"A\"\n".to_owned() + &".a = .a + .a\n".repeat(20);
        let (exceeded, event, result) = run(
            &doubling,
            Value::Object(Default::default()),
            Limits::default(),
        );
        assert_eq!(exceeded, None);
        assert!(result.is_ok());
        assert_eq!(event.get("a").unwrap().as_bytes().unwrap().len(), 1 << 20);
    }

    #[test]
    fn string_doubling_hits_the_memory_limit_before_allocating_past_it() {
        // 30 doublings would be 1 GiB; the cap stops the chain at the first concatenation that
        // would exceed it, so the event keeps the last value that fit.
        let doubling = ".a = \"A\"\n".to_owned() + &".a = .a + .a\n".repeat(30);
        let (exceeded, event, result) =
            run(&doubling, Value::Object(Default::default()), bytes(4 * MIB));
        assert_eq!(exceeded, Some(Limit::Memory));
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("4 MiB memory limit"));
        assert!(event.get("a").unwrap().as_bytes().unwrap().len() <= 2 * MIB);
    }

    #[test]
    fn string_repeat_is_charged_before_it_allocates() {
        let (exceeded, event, _) = run(
            ".a = \"x\" * 4000000000",
            Value::Object(Default::default()),
            bytes(MIB),
        );
        assert_eq!(exceeded, Some(Limit::Memory));
        assert_eq!(event.get("a"), None);

        let (exceeded, _, _) = run(
            ".a = 4000000000 * \"x\"",
            Value::Object(Default::default()),
            bytes(MIB),
        );
        assert_eq!(exceeded, Some(Limit::Memory));
    }

    // Container doublings stop at 14: the type checker's definition of the root doubles per
    // statement too, so compiling 30 of them is itself exponential (that is a compile-time
    // problem, out of scope here).
    #[test]
    fn collection_doubling_hits_the_memory_limit() {
        let doubling = ". = [., .]\n".repeat(14);
        let (exceeded, _, _) = run(
            &doubling,
            Value::Object(Default::default()),
            bytes(64 * 1024),
        );
        assert_eq!(exceeded, Some(Limit::Memory));

        let merging =
            ".a = {\"k\": \"v\"}\n".to_owned() + &".a = {\"l\": .a, \"r\": .a}\n".repeat(14);
        let (exceeded, _, _) = run(
            &merging,
            Value::Object(Default::default()),
            bytes(64 * 1024),
        );
        assert_eq!(exceeded, Some(Limit::Memory));
    }

    #[test]
    fn variable_doubling_hits_the_memory_limit() {
        let doubling = "a = \"A\"\n".to_owned() + &"a = [a, a]\n".repeat(14) + ".a = a";
        let (exceeded, _, _) = run(
            &doubling,
            Value::Object(Default::default()),
            bytes(64 * 1024),
        );
        assert_eq!(exceeded, Some(Limit::Memory));
    }

    #[test]
    fn function_results_are_charged() {
        let growing = ".a = \"A\"\n".to_owned() + &".a = encode_base64(.a + .a)\n".repeat(40);
        let (exceeded, _, _) = run(&growing, Value::Object(Default::default()), bytes(4 * MIB));
        assert_eq!(exceeded, Some(Limit::Memory));
    }

    #[test]
    fn the_limit_is_sticky_through_error_coalescing() {
        // `??` swallows the first error; the next node fails again, and the flag stays set.
        let program = ".a = \"A\"\n".to_owned()
            + &".a = .a + .a\n".repeat(30)
            + ".b = parse_json(.a + .a) ?? \"fallback\"\n.c = \"reached\"";
        let (exceeded, event, result) =
            run(&program, Value::Object(Default::default()), bytes(4 * MIB));
        assert_eq!(exceeded, Some(Limit::Memory));
        assert!(result.is_err());
        assert_eq!(event.get("c"), None);
    }

    #[test]
    fn a_data_dependent_blowup_only_trips_on_the_input_that_triggers_it() {
        let program =
            "if exists(.x) {\n.a = \"A\"\n".to_owned() + &".a = .a + .a\n".repeat(30) + "}";
        let (exceeded, _, result) =
            run(&program, Value::Object(Default::default()), bytes(4 * MIB));
        assert_eq!(exceeded, None);
        assert!(result.is_ok());

        let mut event = Value::Object(Default::default());
        event.insert("x", 1);
        let (exceeded, _, _) = run(&program, event, bytes(4 * MIB));
        assert_eq!(exceeded, Some(Limit::Memory));
    }

    #[test]
    fn the_deadline_stops_a_long_run() {
        // A million-element array mapped through a closure: far past any deadline worth having.
        let mut event = Value::Object(Default::default());
        event.insert("items", Value::Array(vec![Value::Integer(1); 1_000_000]));
        let limits = Limits {
            max_bytes: None,
            max_duration: Some(Duration::from_millis(5)),
        };
        let (exceeded, _, result) = run(
            ".items = map_values(array!(.items)) -> |v| { to_int!(v) + 1 }",
            event,
            limits,
        );
        assert_eq!(exceeded, Some(Limit::Time));
        assert!(result.unwrap_err().to_string().contains("5 ms time limit"));
    }

    #[test]
    fn the_deadline_counts_from_the_start_and_catches_few_but_heavy_nodes() {
        // Fewer than 1024 nodes, each hashing a multi-MB string: the clock is read because the
        // string is large, not because enough nodes went by.
        let mut event = Value::Object(Default::default());
        event.insert("s", Value::Bytes(vec![b'x'; 4 * MIB].into()));
        let limits = Limits {
            max_bytes: None,
            max_duration: Some(Duration::from_millis(5)),
        };
        let program = ".h = sha2(string!(.s))\n".repeat(200);
        let (exceeded, _, _) = run(&program, event, limits);
        assert_eq!(exceeded, Some(Limit::Time));
    }

    #[test]
    fn the_deadline_catches_container_heavy_nodes_with_no_memory_limit() {
        let mut event = Value::Object(Default::default());
        event.insert(
            "items",
            Value::Array((0..200_000).map(Value::Integer).collect()),
        );
        let limits = Limits {
            max_bytes: None,
            max_duration: Some(Duration::from_millis(5)),
        };
        let (exceeded, _, _) = run(&".u = unique(array!(.items))\n".repeat(100), event, limits);
        assert_eq!(exceeded, Some(Limit::Time));
    }

    #[test]
    fn replace_with_is_bounded_as_its_output_grows() {
        let mut event = Value::Object(Default::default());
        event.insert("s", Value::Bytes(vec![b'a'; 1000].into()));
        event.insert("big", Value::Bytes(vec![b'b'; 100 * 1024].into()));
        let (exceeded, event, _) = run(
            ".out = replace_with(string!(.s), r'a') -> |_m| { string!(.big) }",
            event,
            bytes(MIB),
        );
        assert_eq!(exceeded, Some(Limit::Memory));
        assert_eq!(event.get("out"), None);
    }

    #[test]
    fn chained_assignment_clones_are_charged() {
        let mut event = Value::Object(Default::default());
        event.insert("big", Value::Array(vec![Value::Integer(1); 10_000]));
        // Each `=` stores a deep clone of the 10k-element array: 20 of them is 20× the cap.
        let chain = (0..20)
            .map(|i| format!(".c{i}"))
            .collect::<Vec<_>>()
            .join(" = ")
            + " = .big";
        let (exceeded, _, _) = run(&chain, event, bytes(256 * 1024));
        assert_eq!(exceeded, Some(Limit::Memory));
    }

    #[test]
    fn messages_name_the_limit_in_sensible_units() {
        let limits = Limits {
            max_bytes: Some(512 * 1024),
            max_duration: Some(Duration::from_micros(500)),
        };
        assert_eq!(
            limits.message(Limit::Memory),
            "program exceeded the 512 KiB memory limit"
        );
        assert_eq!(
            limits.message(Limit::Time),
            "program exceeded the 0.5 ms time limit"
        );
        assert_eq!(
            bytes(MIB).message(Limit::Memory),
            "program exceeded the 1 MiB memory limit"
        );
    }

    #[test]
    fn ordinary_programs_stay_under_the_limits() {
        let mut event = Value::Object(Default::default());
        event.insert("message", "{\"level\":\"info\",\"user\":{\"id\":7}}");
        let limits = Limits {
            max_bytes: Some(MIB),
            max_duration: Some(Duration::from_millis(5)),
        };
        let (exceeded, event, result) = run(
            ". = merge!(., parse_json!(.message))\n.level = upcase!(.level)\ndel(.message)",
            event,
            limits,
        );
        assert_eq!(exceeded, None);
        assert!(result.is_ok());
        assert_eq!(event.get("level").unwrap(), &Value::from("INFO"));
    }

    fn gzip(bytes: &[u8]) -> Value {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(bytes).unwrap();
        Value::Bytes(encoder.finish().unwrap().into())
    }

    #[test]
    fn decompression_stops_at_the_memory_limit() {
        let mut event = Value::Object(Default::default());
        event.insert("payload", gzip(&vec![0u8; 8 * MIB]));
        let (exceeded, _, _) = run(".out = decode_gzip!(.payload)", event, bytes(MIB));
        assert_eq!(exceeded, Some(Limit::Memory));

        // An output a little under the cap is decoded, i.e. it is not counted twice.
        let mut event = Value::Object(Default::default());
        event.insert("payload", gzip(&vec![0u8; 900 * 1024]));
        let (exceeded, event, _) = run(".out = decode_gzip!(.payload)", event, bytes(MIB));
        assert_eq!(exceeded, None);
        assert_eq!(
            event.get("out").unwrap().as_bytes().unwrap().len(),
            900 * 1024
        );
    }

    #[test]
    fn replace_is_charged_for_its_output_before_building_it() {
        let mut event = Value::Object(Default::default());
        event.insert("s", Value::Bytes(vec![b'a'; 1000].into()));
        event.insert("big", Value::Bytes(vec![b'b'; 100 * 1024].into()));
        for program in [
            ".out = replace(string!(.s), \"a\", string!(.big))",
            ".out = replace(string!(.s), r'a', string!(.big))",
        ] {
            let (exceeded, event, _) = run(program, event.clone(), bytes(MIB));
            assert_eq!(exceeded, Some(Limit::Memory), "{program}");
            assert_eq!(event.get("out"), None, "{program}");
        }
        let (exceeded, event, _) = run(
            ".out = replace(string!(.s), r'a+', \"b\")",
            event,
            bytes(MIB),
        );
        assert_eq!(exceeded, None);
        assert_eq!(event.get("out").unwrap(), &Value::from("b"));
    }

    #[test]
    fn unnest_is_charged_before_copying_the_event() {
        let mut event = Value::Object(Default::default());
        event.insert("items", Value::Array(vec![Value::Integer(1); 1000]));
        event.insert("padding", Value::Array(vec![Value::Integer(1); 1000]));
        let (exceeded, _, _) = run(". = unnest!(.items)", event, bytes(MIB));
        assert_eq!(exceeded, Some(Limit::Memory));
    }

    #[test]
    fn size_estimates() {
        let value = Value::Object(
            [("k".into(), Value::Bytes(vec![b'x'; 1000].into()))]
                .into_iter()
                .collect(),
        );
        assert_eq!(
            clone_size_of(&value),
            OBJECT_BYTES + OBJECT_ENTRY_BYTES + 1 + BYTES_HANDLE
        );
        assert_eq!(
            shallow_size_of(&value),
            OBJECT_BYTES + OBJECT_ENTRY_BYTES + SCALAR_BYTES
        );
        assert_eq!(
            shallow_size_of(&Value::Bytes(vec![b'x'; 1000].into())),
            BYTES_HANDLE + 1000
        );
        let nested = Value::Array(vec![Value::Array(vec![Value::Null; 3]); 2]);
        assert_eq!(
            clone_size_of(&nested),
            ARRAY_BYTES + 2 * (ARRAY_BYTES + 3 * SCALAR_BYTES)
        );
        assert_eq!(shallow_size_of(&nested), ARRAY_BYTES + 2 * SCALAR_BYTES);
    }

    #[test]
    fn remaining_bytes_shrinks_with_charges() {
        let mut budget = Budget::new(bytes(100));
        assert_eq!(budget.remaining_bytes(), 100);
        budget.charge(60).unwrap();
        assert_eq!(budget.remaining_bytes(), 40);
        assert!(budget.charge(41).is_err());
        assert_eq!(budget.remaining_bytes(), 0);
        assert_eq!(budget.exceeded(), Some(Limit::Memory));
        // Still failed afterwards, whatever the size.
        assert!(budget.charge(0).is_err());
        assert!(budget.tick().is_err());
        assert_eq!(Budget::new(Limits::default()).remaining_bytes(), usize::MAX);
    }
}
