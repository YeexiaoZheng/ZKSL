pub type Int = i64;

pub fn exp(x: Int, sf: u64) -> Int {
    let x = x as f64 / sf as f64;
    let exp = x.exp() * sf as f64;
    exp as Int
}

pub fn ln(x: Int, sf: u64) -> Int {
    let x = x as f64 / sf as f64;
    let ln = x.ln() * sf as f64;
    ln as Int
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
