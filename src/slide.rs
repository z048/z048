#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Slide {
    U = 0,
    D = 1,
    L = 2,
    R = 3,
}

impl Slide {
    pub const SLIDES: [Slide; 4] = [Self::U, Self::D, Self::L, Self::R];

    pub fn coord<const N: u8>(self, i: usize, j: usize) -> (usize, usize) {
        let a = if self as u8 & 1 == 0 { j } else { N as usize - 1 - j };
        if self as u8 & 2 == 0 { (a, i) } else { (i, a) }
    }
}

impl From<u16> for Slide {
    fn from(value: u16) -> Self {
        match value {
            0 => Self::U,
            1 => Self::D,
            2 => Self::L,
            3 => Self::R,
            _ => panic!("slide index {value} out of range 0..=3"),
        }
    }
}
