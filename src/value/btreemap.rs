/// A macro to easily create a map containing `Value`
#[macro_export]
macro_rules! btreemap {
    () => (::indexmap::IndexMap::new());

    // trailing comma case
    ($($key:expr => $value:expr,)+) => (btreemap!($($key => $value),+));

    ($($key:expr => $value:expr),*) => {
        ::indexmap::IndexMap::from([
            $(
                ($key.into(), $value.into()),
            )*
        ])
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_btreemap() {
        use indexmap::IndexMap;

        assert_eq!(btreemap! {}, IndexMap::<usize, usize>::new());

        let mut map = IndexMap::new();
        map.insert(1, "1");
        assert_eq!(btreemap! { 1 => "1" }, map);

        let mut map = IndexMap::new();
        map.insert("1", "one");
        map.insert("2", "two");
        assert_eq!(btreemap! { "1" => "one", "2" => "two" }, map);
    }
}
