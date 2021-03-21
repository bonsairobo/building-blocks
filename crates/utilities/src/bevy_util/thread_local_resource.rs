use std::sync::Arc;
use thread_local::ThreadLocal;

#[derive(Default)]
pub struct ThreadLocalResource<T>
where
    T: Send,
{
    tls: Arc<ThreadLocal<T>>,
}

impl<T> ThreadLocalResource<T>
where
    T: Send,
{
    pub fn new() -> Self {
        Self {
            tls: Arc::new(ThreadLocal::new()),
        }
    }

    pub fn get(&self) -> ThreadLocalResourceHandle<T> {
        ThreadLocalResourceHandle {
            tls: self.tls.clone(),
        }
    }

    pub fn into_iter(self) -> impl Iterator<Item = T> {
        let tls = match Arc::try_unwrap(self.tls) {
            Ok(x) => x,
            Err(_) => panic!(
                "Failed to unwrap Arc'd thread-local storage; \
                there must be an outstanding strong reference"
            ),
        };

        tls.into_iter()
    }
}

pub struct ThreadLocalResourceHandle<T>
where
    T: Send,
{
    tls: Arc<ThreadLocal<T>>,
}

impl<T> ThreadLocalResourceHandle<T>
where
    T: Send,
{
    pub fn get_or_create_with(&self, create: impl FnOnce() -> T) -> &T {
        self.tls.get_or(create)
    }

    pub fn get_or_default(&self) -> &T
    where
        T: Default,
    {
        self.tls.get_or_default()
    }
}
