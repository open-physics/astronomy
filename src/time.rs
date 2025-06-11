use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Time(Decimal);

impl Time {
    pub fn from_gps_seconds(seconds: f64) -> Self {
        Time(Decimal::from_f64(seconds).expect("Failed to convert f64 to Decimal for Time."))
    }

    pub fn as_gps_seconds_f64(&self) -> f64 {
        self.0
            .to_f64()
            .expect("Failed to convert Decimal to f64 for Time.")
    }
}
