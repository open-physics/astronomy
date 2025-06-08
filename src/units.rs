use ndarray::Array1;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub enum Dimension {
    Length,
    Time,
    Mass,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    // Simplified representation (e.g. "metre", "second")
    name: String,
    // Scale factor relative to SI (e.g., 0.01 for "centimetre")
    scale: f64,
    // Dimension of the unit (e.g., Length, Time, Mass)
    dimension: Dimension,
}

impl Unit {
    pub fn new(name: &str, scale: f64, dimension: Dimension) -> Self {
        Unit {
            name: name.to_string(),
            scale,
            dimension,
        }
    }
    pub fn is_equivalent(&self, other: &Unit) -> bool {
        // In reality, this would check dimensional equivalence
        self.dimension == other.dimension
    }
}

#[derive(Debug, Error)]
pub enum QuantityError {
    #[error("Incompatible units: {0}")]
    IncompatibleUnits(String),
}

#[derive(Debug, Clone)]
pub struct Quantity {
    value: Array1<f64>,
    unit: Unit,
}

impl Quantity {
    pub fn new(value: Array1<f64>, unit: Unit) -> Self {
        Quantity { value, unit }
    }
    pub fn to(&self, target_unit: &Unit) -> Result<Self, QuantityError> {
        if !self.unit.is_equivalent(target_unit) {
            return Err(QuantityError::IncompatibleUnits(format!(
                "Cannot convert {} to {}",
                self.unit.name, target_unit.name
            )));
        }
        let scale_factor = self.unit.scale / target_unit.scale;
        let new_value = &self.value * scale_factor;
        Ok(Quantity::new(new_value, target_unit.clone()))
    }
}

// Implement basic arithmetic operations for Quantity
impl std::ops::Add for Quantity {
    type Output = Result<Self, QuantityError>;
    fn add(self, rhs: Self) -> Self::Output {
        if self.unit != rhs.unit {
            return Err(QuantityError::IncompatibleUnits(format!(
                "Cannot add {} and {}",
                self.unit.name, rhs.unit.name
            )));
        }
        Ok(Quantity::new(&self.value + &rhs.value, self.unit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_quantity_conversion() {
        let m = Unit::new("m", 1.0, Dimension::Length);
        let cm = Unit::new("cm", 0.01, Dimension::Length);
        let q = Quantity::new(array![1.0, 2.0, 3.0], m);
        let q_cm = q.to(&cm).unwrap();
        assert_eq!(q_cm.value, array![100.0, 200.0, 300.0]);
        // assert_eq!(q_cm.unit.name, "cm");
    }
    #[test]
    fn test_quantity_addition() {
        let unit = Unit::new("m", 1.0, Dimension::Length);
        let q1 = Quantity::new(array![1.0, 2.0], unit.clone());
        let q2 = Quantity::new(array![3.0, 4.0], unit);
        let sum = (q1 + q2).unwrap();
        assert_eq!(sum.value, array![4.0, 6.0]);
    }
}
