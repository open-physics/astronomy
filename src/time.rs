use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use std::fmt::{self, Display};
use std::ops::Add;

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

// Implementing the Display trait for Time
// This is useful for printing Time instances.
impl Display for Time {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Add for Time {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Time(self.0 + other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_from_gps_seconds() {
        let time = Time::from_gps_seconds(123456.789);
        assert_eq!(time.as_gps_seconds_f64(), 123456.789);
    }

    #[test]
    fn test_time_display() {
        let time = Time::from_gps_seconds(98765.4321);
        assert_eq!(format!("{}", time), "98765.4321");
    }

    #[test]
    fn test_time_equality() {
        let time1 = Time::from_gps_seconds(100.0);
        let time2 = Time::from_gps_seconds(100.0);
        let time3 = Time::from_gps_seconds(200.0);

        assert_eq!(time1, time2);
        assert_ne!(time1, time3);
        assert!(time1 < time3);
    }

    #[test]
    fn test_time_addition() {
        let time1 = Time::from_gps_seconds(50.0);
        let time2 = Time::from_gps_seconds(25.0);
        let result = time1 + time2;

        assert_eq!(result.as_gps_seconds_f64(), 75.0);
    }

    #[test]
    fn test_time_ordering() {
        let time1 = Time::from_gps_seconds(300.0);
        let time2 = Time::from_gps_seconds(400.0);
        let time3 = Time::from_gps_seconds(300.0);

        assert!(time1 < time2);
        assert!(time1 <= time3);
        assert!(time2 > time1);
        assert!(time2 >= time3);
    }

    #[test]
    fn test_time_conversion() {
        let time = Time::from_gps_seconds(123456.789);
        let gps_seconds = time.as_gps_seconds_f64();
        assert_eq!(gps_seconds, 123456.789);
    }

    #[test]
    fn test_time_from_decimal() {
        let decimal_time = Decimal::from_f64(123456.789).expect("Failed to create Decimal");
        let time = Time(decimal_time);
        assert_eq!(time.as_gps_seconds_f64(), 123456.789);
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_time_creation_and_conversion() {
        let gps_time = 1126259446.0;
        let time_obj = Time::from_gps_seconds(gps_time);
        assert_eq!(time_obj.as_gps_seconds_f64(), gps_time);

        let precise_gps_time = 123456789.123456789;
        let time_obj_precise = Time::from_gps_seconds(precise_gps_time);
        // Due to f64 precision, we might not get the exact value back
        // For Decimal, it shoiuld be exact

        assert_eq!(
            time_obj_precise.0,
            Decimal::from_f64(precise_gps_time).unwrap()
        );
    }
}
