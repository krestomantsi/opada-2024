fn main() {
    let pantry = vec![Food::Fasolakia(3), Food::Burger, Food::Gyros];
    let yummy_foods = yummy(&pantry);
    println!("{:?}", yummy_foods);
}

// returns if a food is yummy or not
fn yummy(lista: &Vec<Food>) -> Vec<bool> {
    let is_yummy = lista.iter().map(|food| match food {
        Food::Fasolakia(_) => false,
        Food::Burger => true,
        //Food::Gyros => true,
    });
    is_yummy.collect()
}

#[derive(PartialEq)]
enum Food {
    Fasolakia(u16),
    Burger,
    Gyros,
}
