use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::ops::Deref;
use std::ops::DerefMut;

pub struct Dicer {
    v: SmallRng,
}

impl Dicer {
    pub fn softmax<T: Copy>(&mut self, mut options: Vec<(T, f64)>, tau: f64) -> T {
        let mx = options.iter().map(|&(_, s)| s).fold(f64::NEG_INFINITY, f64::max);
        for (_, s) in &mut options {
            *s = if tau == 0.0 {
                (*s == mx) as u8 as f64
            } else if tau.is_infinite() {
                1.0
            } else {
                ((*s - mx) / tau).exp()
            };
        }
        let total: f64 = options.iter().map(|&(_, w)| w).sum();
        let mut x = self.random_range(0.0..total);
        for &(item, w) in &options {
            x -= w;
            if x <= 0.0 {
                return item;
            }
        }
        options.last().expect("softmax called with at least one option").0
    }
}

impl From<u64> for Dicer {
    fn from(state: u64) -> Self {
        SmallRng::seed_from_u64(state).into()
    }
}

impl From<SmallRng> for Dicer {
    fn from(v: SmallRng) -> Self {
        Self { v }
    }
}

impl Deref for Dicer {
    type Target = SmallRng;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl DerefMut for Dicer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}
