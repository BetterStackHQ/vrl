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

        let empty: IndexMap<usize, usize> = btreemap! {};
        assert_eq!(empty, IndexMap::<usize, usize>::new());

        let mut map = IndexMap::new();
        map.insert(1, "1");
        let built: IndexMap<i32, &str> = btreemap! { 1 => "1" };
        assert_eq!(built, map);

        let mut map = IndexMap::new();
        map.insert("1", "one");
        map.insert("2", "two");
        let built: IndexMap<&str, &str> = btreemap! { "1" => "one", "2" => "two" };
        assert_eq!(built, map);
    }
}
