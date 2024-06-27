pub type Int = i64;

pub fn exp(x: Int) -> Int {
    (x as f64).exp() as Int
}

pub fn ln(x: Int) -> Int {
    (x as f64).ln() as Int
}

pub fn sqrt(x: Int) -> Int {
    (x as f64).sqrt() as Int
}

pub fn pow(x: Int, y: Int) -> Int {
    (x as f64).powf(y as f64) as Int
}

pub fn abs(x: Int) -> Int {
    x.abs()
}

pub fn relu(x: Int) -> Int {
    x.max(0)
}
