/// Escapes the capturing of output from tests so it's visible even when the test succeeds.
pub fn test_print(message: &str) {
    use std::io::Write;

    std::io::stdout()
        .lock()
        .write_all(message.as_bytes())
        .unwrap();
}
