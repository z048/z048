use crate::Dicer;
use crate::Slide;
use crate::Spawn;
use rand::Rng;
use rand::seq::IteratorRandom;
use std::ops::DerefMut;
use std::ops::Index;
use std::ops::IndexMut;

#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Board {
    v: [[u8; 4]; 4],
}

impl Board {
    pub fn score(self) -> f64 {
        self.escore().log2()
    }

    pub fn escore(self) -> f64 {
        self.v.into_iter().flatten().map(|v| v as f64).map(|v| v * v.exp2()).sum()
    }

    pub fn end(self) -> bool {
        for x in 0..4 {
            for y in 0..4 {
                let c = self.v[x][y];
                if c == 0 || (x < 3 && self.v[x + 1][y] == c) || (y < 3 && self.v[x][y + 1] == c) {
                    return false;
                }
            }
        }
        true
    }

    pub fn empties(self) -> impl Iterator<Item = (usize, usize)> {
        (0..4).flat_map(|x| (0..4).map(move |y| (x, y))).filter(move |&c| self.is_empty(c))
    }

    pub fn is_empty(self, (x, y): (usize, usize)) -> bool {
        self.v[x][y] == 0
    }

    pub fn is_legal_slide(self, s: Slide) -> bool {
        for i in 0..4 {
            let mut hole = false;
            for p in 0..4 {
                let v = self[s.coord::<4>(i, p)];
                if v == 0 {
                    hole = true;
                } else if hole || (p < 3 && self[s.coord::<4>(i, p + 1)] == v) {
                    return true;
                }
            }
        }
        false
    }

    pub fn is_legal_spawn(self, s: Spawn<4, 2>) -> bool {
        self.is_empty(s.xy())
    }

    pub fn iter_legal_slide(self) -> impl Iterator<Item = Slide> {
        Slide::SLIDES.into_iter().filter(move |&s| self.is_legal_slide(s))
    }

    pub fn iter_legal_spawn(self) -> impl Iterator<Item = Spawn<4, 2>> {
        self.empties().flat_map(|c| [(c, 1).into(), (c, 2).into()])
    }

    pub fn slide(mut self, s: Slide) -> Self {
        for i in 0..4 {
            let mut line = [0u8; 4];
            let mut n = 0;
            for p in 0..4 {
                let val = self[s.coord::<4>(i, p)];
                if val != 0 {
                    line[n] = val;
                    n += 1;
                }
            }
            let (mut w, mut k) = (0, 0);
            while k < n {
                let pair = k + 1 < n && line[k] == line[k + 1];
                line[w] = line[k] + pair as u8;
                k += 1 + pair as usize;
                w += 1;
            }
            line[w..].fill(0);
            for p in 0..4 {
                self[s.coord::<4>(i, p)] = line[p as usize];
            }
        }
        self
    }

    pub fn spawn(mut self, s: Spawn<4, 2>) -> Self {
        let ((x, y), n) = s.cm();
        self.v[x][y] = n;
        self
    }

    pub fn symmetries(self) -> [Self; 8] {
        fn r0(x: usize, y: usize) -> (usize, usize) {
            (x, y)
        }
        fn r1(x: usize, y: usize) -> (usize, usize) {
            (3 - y, x)
        }
        fn r2(x: usize, y: usize) -> (usize, usize) {
            (3 - x, 3 - y)
        }
        fn r3(x: usize, y: usize) -> (usize, usize) {
            (y, 3 - x)
        }
        fn f0(x: usize, y: usize) -> (usize, usize) {
            (x, 3 - y)
        }
        fn f1(x: usize, y: usize) -> (usize, usize) {
            (3 - x, y)
        }
        fn f2(x: usize, y: usize) -> (usize, usize) {
            (y, x)
        }
        fn f3(x: usize, y: usize) -> (usize, usize) {
            (3 - y, 3 - x)
        }
        [r0, r1, r2, r3, f0, f1, f2, f3].map(|f| {
            let mut ret = self;
            for x in 0..4 {
                for y in 0..4 {
                    ret[(x, y)] = self[f(x, y)];
                }
            }
            ret
        })
    }
}

impl From<u64> for Board {
    fn from(value: u64) -> Self {
        let mut dicer = Dicer::from(value);
        let mut board = Self::default();
        for c in (0..16).choose_multiple(dicer.deref_mut(), 2) {
            board = board.spawn((c, dicer.random_range(1..=2u8)).into());
        }
        board
    }
}

impl From<[[u8; 4]; 4]> for Board {
    fn from(v: [[u8; 4]; 4]) -> Self {
        Board { v }
    }
}

impl From<Board> for [[u8; 4]; 4] {
    fn from(Board { v }: Board) -> Self {
        v
    }
}

impl Index<(usize, usize)> for Board {
    type Output = u8;
    fn index(&self, (x, y): (usize, usize)) -> &u8 {
        &self.v[x][y]
    }
}

impl IndexMut<(usize, usize)> for Board {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        &mut self.v[x][y]
    }
}
