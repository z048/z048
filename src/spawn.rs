#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct Spawn<const N: u8, const M: u8> {
    v: u16,
}

impl<const N: u8, const M: u8> Spawn<N, M> {
    pub fn cm(self) -> ((usize, usize), u8) {
        let m = self.v / (N as u16 * N as u16) + 1;
        let c = self.v % (N as u16 * N as u16);
        let x = c / N as u16;
        let y = c % N as u16;
        ((x as usize, y as usize), m as u8)
    }

    pub fn xy(self) -> (usize, usize) {
        let c = self.v % (N as u16 * N as u16);
        let x = c / N as u16;
        let y = c % N as u16;
        (x as usize, y as usize)
    }
}

impl<const N: u8, const M: u8> From<u16> for Spawn<N, M> {
    fn from(v: u16) -> Self {
        Self { v }
    }
}

impl<const N: u8, const M: u8> From<(u8, u8)> for Spawn<N, M> {
    fn from((c, m): (u8, u8)) -> Self {
        assert!((0..N * N).contains(&c));
        assert!((1..=M).contains(&m));
        ((m as u16 - 1) * N as u16 * N as u16 + c as u16).into()
    }
}

impl<const N: u8, const M: u8> From<((usize, usize), u8)> for Spawn<N, M> {
    fn from(((x, y), m): ((usize, usize), u8)) -> Self {
        assert!((0..N as usize).contains(&x));
        assert!((0..N as usize).contains(&y));
        (x as u8 * N + y as u8, m).into()
    }
}

impl<const N: u8, const M: u8> From<Spawn<N, M>> for u16 {
    fn from(Spawn { v }: Spawn<N, M>) -> Self {
        v
    }
}
