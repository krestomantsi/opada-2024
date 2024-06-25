fn main() {
    let x = vec![1.0f32, 2.0, 3.0];
    let y = vec![4.0f32, 5.0, 6.0];
    let z = zero(x);
    let w = zero(x);
    println!("Hurray!!");
}

fn zero(x: Vec<f32>) -> Vec<f32> {
    x.iter().map(|a| a * 0.0).collect::<Vec<f32>>()
}
