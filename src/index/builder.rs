use crate::error::Error;
use crate::index::external;
use crate::index::key::Indexable;
#[cfg(feature = "std")]
use crate::index::owned;

/// Builder for constructing PGM indices with custom parameters.
///
/// # Example
///
/// ```
/// use pgm_extra::index::Builder;
///
/// let data: Vec<u64> = (0..10000).collect();
///
/// let index = Builder::new()
///     .epsilon(128)
///     .epsilon_recursive(8)
///     .build(&data)
///     .unwrap();
///
/// assert_eq!(index.epsilon(), 128);
/// ```
#[derive(Clone, Debug)]
pub struct Builder {
    epsilon: usize,
    epsilon_recursive: usize,
    #[cfg(feature = "parallel")]
    parallel: bool,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            epsilon: 64,
            epsilon_recursive: 4,
            #[cfg(feature = "parallel")]
            parallel: false,
        }
    }
}

impl Builder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the epsilon (error bound) for the bottom level.
    pub fn epsilon(mut self, epsilon: usize) -> Self {
        self.epsilon = epsilon.max(1);
        self
    }

    /// Set the epsilon (error bound) for the upper levels.
    pub fn epsilon_recursive(mut self, epsilon_recursive: usize) -> Self {
        self.epsilon_recursive = epsilon_recursive;
        self
    }

    #[cfg(feature = "parallel")]
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn build<T: Indexable>(&self, data: &[T]) -> Result<external::Static<T>, Error>
    where
        T::Key: Ord,
    {
        #[cfg(feature = "parallel")]
        {
            if self.parallel {
                return external::Static::new_parallel(data, self.epsilon, self.epsilon_recursive);
            }
        }

        external::Static::new(data, self.epsilon, self.epsilon_recursive)
    }

    pub fn build_one_level<T: Indexable>(&self, data: &[T]) -> Result<external::OneLevel<T>, Error>
    where
        T::Key: Ord,
    {
        external::OneLevel::new(data, self.epsilon)
    }

    #[cfg(feature = "std")]
    pub fn build_dynamic<T: Indexable + Ord + Copy>(
        &self,
        data: Vec<T>,
    ) -> Result<owned::Dynamic<T>, Error>
    where
        T::Key: Ord,
    {
        owned::Dynamic::from_sorted(data, self.epsilon, self.epsilon_recursive)
    }

    #[cfg(feature = "std")]
    pub fn build_dynamic_empty<T: Indexable + Ord + Copy>(&self) -> owned::Dynamic<T>
    where
        T::Key: Ord,
    {
        owned::Dynamic::new(self.epsilon, self.epsilon_recursive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn test_builder_default() {
        let builder = Builder::new();
        let data: Vec<u64> = (0..1000).collect();
        let index = builder.build(&data).unwrap();

        assert_eq!(index.epsilon(), 64);
    }

    #[test]
    fn test_builder_custom_epsilon() {
        let builder = Builder::new().epsilon(128).epsilon_recursive(8);
        let data: Vec<u64> = (0..1000).collect();
        let index = builder.build(&data).unwrap();

        assert_eq!(index.epsilon(), 128);
        assert_eq!(index.epsilon_recursive(), 8);
    }

    #[test]
    fn test_builder_one_level() {
        let builder = Builder::new().epsilon(32);
        let data: Vec<u64> = (0..1000).collect();
        let index = builder.build_one_level(&data).unwrap();

        assert_eq!(index.epsilon(), 32);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_builder_dynamic() {
        let builder = Builder::new();
        let data: Vec<u64> = (0..100).collect();
        let index = builder.build_dynamic(data).unwrap();

        assert_eq!(index.len(), 100);
    }
}
