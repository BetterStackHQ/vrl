use super::limits::{self, Budget, Limit, Limits};
use super::ExpressionError;
use super::TimeZone;
use crate::value::Value;

use super::{state::RuntimeState, Target};

pub struct Context<'a> {
    target: &'a mut dyn Target,
    state: &'a mut RuntimeState,
    timezone: &'a TimeZone,
    budget: Budget,
}

impl<'a> Context<'a> {
    /// Create a new [`Context`] with no resource limits.
    pub fn new(
        target: &'a mut dyn Target,
        state: &'a mut RuntimeState,
        timezone: &'a TimeZone,
    ) -> Self {
        Self::with_limits(target, state, timezone, Limits::default())
    }

    /// Create a new [`Context`] whose run is subject to `limits` (see [`limits`]).
    pub fn with_limits(
        target: &'a mut dyn Target,
        state: &'a mut RuntimeState,
        timezone: &'a TimeZone,
        limits: Limits,
    ) -> Self {
        Self {
            target,
            state,
            timezone,
            budget: Budget::new(limits),
        }
    }

    /// Get a reference to the [`Target`].
    #[must_use]
    pub fn target(&self) -> &dyn Target {
        self.target
    }

    /// Get a mutable reference to the [`Target`].
    pub fn target_mut(&mut self) -> &mut dyn Target {
        self.target
    }

    /// Get a reference to the [`runtime state`](Runtime).
    #[must_use]
    pub fn state(&self) -> &RuntimeState {
        self.state
    }

    /// Get a mutable reference to the [`runtime state`](Runtime).
    pub fn state_mut(&mut self) -> &mut RuntimeState {
        self.state
    }

    /// Get a reference to the [`TimeZone`]
    #[must_use]
    pub fn timezone(&self) -> &TimeZone {
        self.timezone
    }

    /// Count one expression node against the time limit. Called by every `Expr::resolve`.
    #[inline]
    pub fn tick(&mut self) -> Result<(), ExpressionError> {
        self.budget.tick()
    }

    /// Account for `bytes` of value the program is about to build, or has just built.
    #[inline]
    pub fn charge(&mut self, bytes: usize) -> Result<(), ExpressionError> {
        self.budget.charge(bytes)
    }

    /// Account for a value a function or operator just built: its string buffer at full length,
    /// a container by its top-level element count (see [`limits::shallow_size_of`]).
    #[inline]
    pub fn charge_result(&mut self, value: &Value) -> Result<(), ExpressionError> {
        self.note_input(value);
        if self.budget.has_memory_limit() {
            return self.budget.charge(limits::shallow_size_of(value));
        }
        Ok(())
    }

    /// Account for a clone of `value` (string buffers are shared, so only their handle counts).
    #[inline]
    pub fn charge_clone(&mut self, value: &Value) -> Result<(), ExpressionError> {
        self.note_input(value);
        if self.budget.has_memory_limit() {
            return self.budget.charge(limits::clone_size_of(value));
        }
        Ok(())
    }

    /// A large value flowing through a node makes the next tick read the clock, whatever it was
    /// charged: the nodes that take long are the ones working on a lot of bytes or elements. A
    /// container is judged by its top-level length (O(1)), at `SCALAR_BYTES` per element.
    #[inline]
    fn note_input(&mut self, value: &Value) {
        match value {
            Value::Bytes(bytes) => self.budget.note_input(bytes.len()),
            Value::Array(items) => self.budget.note_input(items.len() * limits::SCALAR_BYTES),
            Value::Object(map) => self.budget.note_input(map.len() * limits::SCALAR_BYTES),
            _ => {}
        }
    }

    /// Whether the run has a memory limit at all, for sites that would otherwise pay for an
    /// estimate nobody reads.
    #[must_use]
    pub fn has_memory_limit(&self) -> bool {
        self.budget.has_memory_limit()
    }

    /// Bytes the program may still build; `usize::MAX` without a memory limit. For sites that
    /// must bound an allocation before making it (decoders, `replace`).
    #[must_use]
    pub fn remaining_bytes(&self) -> usize {
        self.budget.remaining_bytes()
    }

    /// Fail the run if a value of `bytes` would not fit in what is left of the memory limit. For
    /// a function that knows (or bounds) its output size before building it; the output itself is
    /// charged by `FunctionCall` once it exists, so this does not charge.
    #[inline]
    pub fn ensure_fits(&mut self, bytes: usize) -> Result<(), ExpressionError> {
        if bytes > self.budget.remaining_bytes() {
            return Err(self.budget.exceed(Limit::Memory));
        }
        Ok(())
    }

    /// Mark the run as over `limit` and return the error to raise.
    pub fn exceed(&mut self, limit: Limit) -> ExpressionError {
        self.budget.exceed(limit)
    }

    /// The limit this run exceeded, if any. Sticky, so a program that swallowed the error with
    /// `??` or `, err =` still reports it.
    #[must_use]
    pub fn exceeded(&self) -> Option<Limit> {
        self.budget.exceeded()
    }
}
