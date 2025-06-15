use ndarray::Array1;
use thiserror::Error;

// Define the base dimensions of physical quantities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dimension {
    Length,              // L - e.g., meters (m), centimeters (cm), kilometers (km)
    Mass,                // M - e.g., kilograms (kg), grams (g)
    Time,                // T - e.g., seconds (s), milliseconds (ms)
    ElectricCurrent,     // I - e.g., amperes (A), milliamperes (mA)
    AbsoluteTemperature, // Theta - e.g., kelvin (K), celsius (C)
    AmountOfSubstance,   // N - e.g., moles (mol)
    LuminousIntensity,   // J - e.g., candela (cd)
}

// Implement Unit Multiplication/Division
#[derive(Debug, Clone, PartialEq)]
pub struct UnitProduct {
    // (Dimension, exponent)
    // e.g., [(Dimension::Length, 1), (Dimension::Time, -1)] for velocity
    components: [(Dimension, i32); 7],
}
// Implement methods for UnitProduct
impl UnitProduct {
    pub const fn new(dimension: Dimension) -> Self {
        // Initialize an array with all exponents as 0
        let mut components = [
            (Dimension::Length, 0),
            (Dimension::Mass, 0),
            (Dimension::Time, 0),
            (Dimension::ElectricCurrent, 0),
            (Dimension::AbsoluteTemperature, 0),
            (Dimension::AmountOfSubstance, 0),
            (Dimension::LuminousIntensity, 0),
        ];
        let mut i = 0;
        while i < components.len() {
            if components[i].0 as usize == dimension as usize {
                components[i].1 += 1; // Increment the exponent for the specified dimension
                break;
            }
            i += 1;
        }
        UnitProduct { components }
    }
    pub const fn from_components(dims: &[(Dimension, i32)]) -> Self {
        let mut components = [
            (Dimension::Length, 0),
            (Dimension::Mass, 0),
            (Dimension::Time, 0),
            (Dimension::ElectricCurrent, 0),
            (Dimension::AbsoluteTemperature, 0),
            (Dimension::AmountOfSubstance, 0),
            (Dimension::LuminousIntensity, 0),
        ];
        let mut i = 0;
        while i < dims.len() {
            let (dim, exp) = dims[i];
            let mut j = 0;
            while j < components.len() {
                if components[j].0 as usize == dim as usize {
                    components[j].1 = exp;
                    break;
                }
                j += 1;
            }
            i += 1;
        }
        UnitProduct { components }
    }
    pub fn multiply(&self, other: &Self) -> Self {
        let mut result_components = self.components;
        for (other_dim, other_exp) in &other.components {
            for (dim, exp) in result_components.iter_mut() {
                if dim == other_dim {
                    *exp += other_exp; // Add the exponent if dimension matches
                    break;
                }
            }
        }
        UnitProduct {
            components: result_components,
        }
    }

    // Helper for division (or negative exponents) in a constant context
    pub const fn inverse(mut self) -> Self {
        let mut i = 0;
        while i < self.components.len() {
            self.components[i].1 *= -1;
            i += 1;
        }
        self
    }
}

// Define a unit of measurement
#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    // Simplified representation (e.g. "metre", "second")
    pub name: &'static str,
    // Scale factor relative to SI (e.g., 0.01 for "centimetre")
    pub scale: f64,
    // Dimension of the unit (e.g., Length, Time, Mass)
    pub dimensions: UnitProduct,
}

impl Unit {
    pub fn new(name: &'static str, scale: f64, dimensions: UnitProduct) -> Self {
        Unit {
            name,
            scale,
            dimensions,
        }
    }
    // Check if two units are equivalent based on their dimensions
    pub fn is_equivalent(&self, other: &Unit) -> bool {
        self.dimensions == other.dimensions
    }
}
// Predefine some common units
pub const SECOND: Unit = Unit {
    name: "s",
    scale: 1.0,
    dimensions: UnitProduct::new(Dimension::Time),
};
pub const METER: Unit = Unit {
    name: "m",
    scale: 1.0,
    dimensions: UnitProduct::new(Dimension::Length),
};
pub const KILOGRAM: Unit = Unit {
    name: "kg",
    scale: 1.0,
    dimensions: UnitProduct::new(Dimension::Mass),
};
pub const AMPERE: Unit = Unit {
    name: "A",
    scale: 1.0,
    dimensions: UnitProduct::new(Dimension::ElectricCurrent),
};
pub const KELVIN: Unit = Unit {
    name: "K",
    scale: 1.0,
    dimensions: UnitProduct::new(Dimension::AbsoluteTemperature),
};
pub const MOLE: Unit = Unit {
    name: "mol",
    scale: 1.0,
    dimensions: UnitProduct::new(Dimension::AmountOfSubstance),
};
pub const CANDELA: Unit = Unit {
    name: "cd",
    scale: 1.0,
    dimensions: UnitProduct::new(Dimension::LuminousIntensity),
};
pub const CENTIMETER: Unit = Unit {
    name: "cm",
    scale: 0.01,
    dimensions: UnitProduct::new(Dimension::Length),
};

// Derived units can be created by combining base units
pub const METER_PER_SECOND: Unit = Unit {
    name: "m/s",
    scale: 1.0,
    dimensions: UnitProduct::from_components(&[(Dimension::Length, 1), (Dimension::Time, -1)]),
};

pub const NEWTON: Unit = Unit {
    name: "N",
    scale: 1.0,
    dimensions: UnitProduct::from_components(&[
        (Dimension::Mass, 1),
        (Dimension::Length, 1),
        (Dimension::Time, -2),
    ]),
};

pub const JOULE: Unit = Unit {
    name: "J",
    scale: 1.0,
    dimensions: UnitProduct::from_components(&[
        (Dimension::Mass, 1),
        (Dimension::Length, 2),
        (Dimension::Time, -2),
    ]),
};

// --- Quantities ---

#[derive(Debug, Error)]
pub enum QuantityError {
    #[error("Incompatible units: Cannot convert '{from}' to '{to}'. Dimensions differ.")]
    IncompatibleUnits { from: String, to: String },
    #[error("Incompatible units: Cannot add '{lhs}' and '{rhs}'. Units are not identical.")]
    IncompatibleAddition { lhs: String, rhs: String },
    #[error("Incompatible units: Cannot subtract '{lhs}' and '{rhs}'. Units are not identical.")]
    IncompatibleSubtraction { lhs: String, rhs: String },
    #[error("Incompatible units: Cannot multiply '{lhs}' and '{rhs}'. Units are not compatible.")]
    IncompatibleMultiplication { lhs: String, rhs: String },
    #[error("Incompatible units: Cannot divide '{lhs}' by '{rhs}'. Units are not compatible.")]
    IncompatibleDivision { lhs: String, rhs: String },
    #[error("Invalid unit: '{0}'")]
    InvalidUnit(String),
    #[error("Invalid quantity: '{0}'")]
    InvalidQuantity(String),
    #[error("Invalid operation: Cannot divide by zero.")]
    DivideByZero,
    #[error("Mismatched quantity: {0}")]
    MismatchError(String),
}

// Define Quantity type to represent physical quantities with units
#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    pub value: Array1<f64>,
    pub unit: Unit,
}

impl Quantity {
    pub fn new(value: Array1<f64>, unit: Unit) -> Self {
        Quantity { value, unit }
    }

    // Convert the Quantity to a target unit if dimensions match
    pub fn to(&self, target_unit: &Unit) -> Result<Self, QuantityError> {
        // Use the is_equivalent method to check if units are compatible
        if !self.unit.is_equivalent(target_unit) {
            return Err(QuantityError::IncompatibleUnits {
                from: self.unit.name.to_string(),
                to: target_unit.name.to_string(),
            });
        }
        let scale_factor = self.unit.scale / target_unit.scale;
        let new_value = &self.value * scale_factor;
        Ok(Quantity::new(new_value, target_unit.clone()))
    }

    // Implement basic arithmetic operations for Quantity
    // Multiply the Quantity by a scalar
    pub fn multiply_scalar(&self, scalar: f64) -> Result<Self, QuantityError> {
        Ok(Quantity::new(&self.value * scalar, self.unit.clone()))
    }
    pub fn divide_scalar(&self, scalar: f64) -> Result<Self, QuantityError> {
        if scalar == 0.0 {
            return Err(QuantityError::DivideByZero);
        }
        Ok(Quantity::new(&self.value / scalar, self.unit.clone()))
    }
}

// Implement basic arithmetic operations for Quantity
use std::ops::{Add, Div, Mul, Sub};

// Implement addition for Quantity
impl Add for Quantity {
    type Output = Result<Self, QuantityError>;
    fn add(self, rhs: Self) -> Self::Output {
        if self.unit != rhs.unit {
            return Err(QuantityError::IncompatibleAddition {
                lhs: self.unit.name.to_string(),
                rhs: rhs.unit.name.to_string(),
            });
        }
        Ok(Quantity::new(&self.value + &rhs.value, self.unit))
    }
}

// Implement subtraction for Quantity
impl Sub for Quantity {
    type Output = Result<Self, QuantityError>;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.unit != rhs.unit {
            return Err(QuantityError::IncompatibleSubtraction {
                lhs: self.unit.name.to_string(),
                rhs: rhs.unit.name.to_string(),
            });
        }
        Ok(Quantity::new(&self.value - &rhs.value, self.unit))
    }
}

// Implement multiplication for Quantity
impl Mul for Quantity {
    type Output = Self; // Multiplication of two quantities results in a new quantity with a new unit
    fn mul(self, rhs: Self) -> Self::Output {
        let new_value = &self.value * &rhs.value;
        let new_unit_dimensions = self.unit.dimensions.multiply(&rhs.unit.dimensions);
        let new_unit_name = format!("{}*{}", self.unit.name, rhs.unit.name);
        let new_unit_scale = self.unit.scale * rhs.unit.scale;

        Quantity::new(
            new_value,
            Unit {
                name: new_unit_name.leak(), // Convert to static str
                scale: new_unit_scale,
                dimensions: new_unit_dimensions,
            },
        )
    }
}

// Implement division for Quantity
impl Div for Quantity {
    type Output = Result<Self, QuantityError>;
    fn div(self, rhs: Self) -> Self::Output {
        // Check for division by zero values in the rhs Array1
        if rhs.value.iter().any(|&x| x == 0.0) {
            return Err(QuantityError::DivideByZero);
        }

        let new_value = &self.value / &rhs.value;
        let new_unit_dimensions = self
            .unit
            .dimensions
            .multiply(&rhs.unit.dimensions.inverse());
        let new_unit_name = format!("{}/{}", self.unit.name, rhs.unit.name);
        let new_unit_scale = self.unit.scale / rhs.unit.scale;

        Ok(Quantity::new(
            new_value,
            Unit {
                name: new_unit_name.leak(), // Convert to static str
                scale: new_unit_scale,
                dimensions: new_unit_dimensions,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_quantity_new() {
        let q = Quantity::new(array![1.0], METER.clone());
        assert_eq!(q.value, array![1.0]);
        assert_eq!(q.unit.name, "m");
        assert_eq!(q.unit, METER);
    }

    #[test]
    fn test_quantity_to() {
        let m = METER.clone();
        let cm = CENTIMETER.clone();
        let q = Quantity::new(array![1.0, 2.0, 3.0], m);
        let q_cm = q.to(&cm).unwrap();
        assert_eq!(q_cm.value, array![100.0, 200.0, 300.0]);
        assert_eq!(q_cm.unit.name, "cm");
    }

    #[test]
    fn test_quantity_multiply_scalar() {
        let q = Quantity::new(array![1.0, 2.0], METER.clone());
        let result = q.multiply_scalar(2.0).unwrap();
        assert_eq!(result.value, array![2.0, 4.0]);
        assert_eq!(result.unit.name, "m");
    }
    #[test]
    fn test_quantity_divide_scalar() {
        let q = Quantity::new(array![2.0, 4.0], METER.clone());
        let result = q.divide_scalar(2.0).unwrap();
        assert_eq!(result.value, array![1.0, 2.0]);
        assert_eq!(result.unit.name, "m");
    }
    #[test]
    fn test_quantity_divide_scalar_by_zero() {
        let q = Quantity::new(array![2.0, 4.0], METER.clone());
        let result = q.divide_scalar(0.0);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Invalid operation: Cannot divide by zero."
        );
    }
    #[test]
    fn test_quantity_addition() {
        let unit = METER.clone();
        let q1 = Quantity::new(array![1.0, 2.0], unit.clone());
        let q2 = Quantity::new(array![3.0, 4.0], unit);
        let sum = (q1 + q2).unwrap();
        assert_eq!(sum.value, array![4.0, 6.0]);
        assert_eq!(sum.unit.name, "m");
    }
    #[test]
    fn test_quantity_subtraction() {
        let unit = METER.clone();
        let q1 = Quantity::new(array![5.0, 6.0], unit.clone());
        let q2 = Quantity::new(array![3.0, 4.0], unit);
        let difference = (q1 - q2).unwrap();
        assert_eq!(difference.value, array![2.0, 2.0]);
        assert_eq!(difference.unit.name, "m");
    }
    #[test]
    fn test_quantity_multiplication() {
        let q1 = Quantity::new(array![1.0, 2.0], METER.clone());
        let q2 = Quantity::new(array![3.0, 4.0], SECOND.clone());
        let product = q1 * q2;
        assert_eq!(product.value, array![3.0, 8.0]);
        assert_eq!(product.unit.name, "m*s");
    }
    #[test]
    fn test_quantity_division() {
        let q1 = Quantity::new(array![6.0, 8.0], METER.clone());
        let q2 = Quantity::new(array![2.0, 4.0], SECOND.clone());
        let quotient = q1 / q2;
        assert!(quotient.is_ok());
        let quotient = quotient.unwrap();
        assert_eq!(quotient.value, array![3.0, 2.0]);
        assert_eq!(quotient.unit.name, "m/s");
    }
    #[test]
    fn test_quantity_divide_by_zero() {
        let q1 = Quantity::new(array![6.0, 8.0], METER.clone());
        let q2 = Quantity::new(array![0.0, 4.0], SECOND.clone());
        let result = q1 / q2;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Invalid operation: Cannot divide by zero."
        );
    }

    #[test]
    fn test_quantity_addition_incompatible_units() {
        let q1 = Quantity::new(array![1.0, 2.0], METER.clone());
        let q2 = Quantity::new(array![3.0, 4.0], SECOND.clone());
        let result = q1 + q2;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Incompatible units: Cannot add 'm' and 's'. Units are not identical."
        );
    }
    #[test]
    fn test_quantity_subtraction_incompatible_units() {
        let q1 = Quantity::new(array![5.0, 6.0], METER.clone());
        let q2 = Quantity::new(array![3.0, 4.0], SECOND.clone());
        let result = q1 - q2;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Incompatible units: Cannot subtract 'm' and 's'. Units are not identical."
        );
    }
    #[test]
    fn test_quantity_multiplication_incompatible_units() {
        let q1 = Quantity::new(array![1.0, 2.0], METER.clone());
        let q2 = Quantity::new(array![3.0, 4.0], SECOND.clone());
        let result = q1 * q2;
        assert_eq!(result.value, array![3.0, 8.0]);
        assert_eq!(result.unit.name, "m*s");
    }
    #[test]
    fn test_quantity_division_incompatible_units() {
        let q1 = Quantity::new(array![6.0, 8.0], METER.clone());
        let q2 = Quantity::new(array![2.0, 4.0], SECOND.clone());
        let result = q1 / q2;
        assert!(result.is_ok());
        let quotient = result.unwrap();
        assert_eq!(quotient.value, array![3.0, 2.0]);
        assert_eq!(quotient.unit.name, "m/s");
    }
    #[test]
    fn test_quantity_addition_compatible_units() {
        let q1 = Quantity::new(array![1.0, 2.0], METER.clone());
        let q2 = Quantity::new(array![3.0, 4.0], CENTIMETER.clone());
        let result = q1.to(&CENTIMETER).unwrap() + q2;
        assert!(result.is_ok());
        let sum = result.unwrap();
        assert_eq!(sum.value, array![103.0, 204.0]);
        assert_eq!(sum.unit.name, "cm");
    }
    #[test]
    fn test_quantity_subtraction_compatible_units() {
        let q1 = Quantity::new(array![5.0, 6.0], METER.clone());
        let q2 = Quantity::new(array![3.0, 4.0], CENTIMETER.clone());
        let result = q1.to(&CENTIMETER).unwrap() - q2;
        assert!(result.is_ok());
        let difference = result.unwrap();
        assert_eq!(difference.value, array![497.0, 596.0]);
        assert_eq!(difference.unit.name, "cm");
    }
}
