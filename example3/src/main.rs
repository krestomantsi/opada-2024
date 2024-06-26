// An integer division that doesn't `panic!`
fn checked_division(dividend: i32, divisor: i32) -> Option<i32> {
    if divisor == 0 {
        // Failure is represented as the `None` variant
        None
    } else {
        // Result is wrapped in a `Some` variant
        Some(dividend / divisor)
    }
}

fn main() {
    let x = checked_division(4, 2);
    let y = checked_division(1, 0);

    // Unwrapping a `Some` variant will extract the value wrapped.
    println!("x is {:?}", x.unwrap());

    // error handle
    match y {
        Some(v) => println!("y is {:?}", v),
        None => println!("y is None"),
    }

    // Unwrapping a `None` variant will `panic!`
    println!("y is {:?}", y.unwrap());
}
