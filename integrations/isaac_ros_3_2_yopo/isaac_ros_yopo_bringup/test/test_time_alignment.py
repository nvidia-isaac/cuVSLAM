import math
from types import SimpleNamespace
import unittest

from isaac_ros_yopo_bringup.time_alignment import (
    ConsecutiveFailureGate,
    MAX_TIME_NANOSECONDS,
    StampOrder,
    StrictStampGuard,
    TimestampContractError,
    add_offset_nanoseconds,
    clock_residual_nanoseconds,
    clone_with_aligned_stamp,
    split_nanoseconds,
    stamp_to_nanoseconds,
)


def make_message(sec=10, nanosec=20, frame_id="base_link"):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=sec, nanosec=nanosec),
            frame_id=frame_id,
        ),
        orientation=SimpleNamespace(x=0.1, y=-0.2, z=0.3, w=0.9),
        orientation_covariance=[-1.0, 2.0, math.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        angular_velocity=SimpleNamespace(x=1.1, y=-2.2, z=3.3),
        angular_velocity_covariance=list(range(9)),
        linear_acceleration=SimpleNamespace(x=-4.4, y=5.5, z=9.81),
        linear_acceleration_covariance=[value / 10.0 for value in range(9)],
    )


class TimestampArithmeticTest(unittest.TestCase):
    def test_applies_audited_offset_exactly(self):
        input_ns = stamp_to_nanoseconds(1_784_533_873, 524_838_125)
        output_ns = add_offset_nanoseconds(input_ns, 1_737_987)
        self.assertEqual(1_737_987, output_ns - input_ns)

    def test_nanosecond_carry(self):
        input_ns = stamp_to_nanoseconds(10, 999_999_500)
        self.assertEqual((11, 500), split_nanoseconds(input_ns + 1_000))

    def test_negative_offset_borrows_from_second(self):
        input_ns = stamp_to_nanoseconds(2, 100)
        output_ns = add_offset_nanoseconds(input_ns, -200)
        self.assertEqual((1, 999_999_900), split_nanoseconds(output_ns))

    def test_zero_is_representable_but_negative_is_rejected(self):
        self.assertEqual((0, 0), split_nanoseconds(add_offset_nanoseconds(10, -10)))
        with self.assertRaises(TimestampContractError):
            add_offset_nanoseconds(10, -11)

    def test_rejects_an_invalid_input_even_if_offset_would_recover_it(self):
        with self.assertRaises(TimestampContractError):
            add_offset_nanoseconds(-1, 1)
        with self.assertRaises(TimestampContractError):
            add_offset_nanoseconds(MAX_TIME_NANOSECONDS + 1, -1)

    def test_ros_time_upper_boundary(self):
        self.assertEqual(
            (2_147_483_647, 999_999_999),
            split_nanoseconds(MAX_TIME_NANOSECONDS),
        )
        with self.assertRaises(TimestampContractError):
            add_offset_nanoseconds(MAX_TIME_NANOSECONDS, 1)

    def test_rejects_invalid_nanosecond_field(self):
        self.assertEqual(
            1_999_999_999,
            stamp_to_nanoseconds(1, 999_999_999),
        )
        with self.assertRaises(TimestampContractError):
            stamp_to_nanoseconds(1, 1_000_000_000)

    def test_large_epoch_keeps_single_nanosecond_precision(self):
        first = stamp_to_nanoseconds(1_784_533_873, 999_999_998)
        second = stamp_to_nanoseconds(1_784_533_873, 999_999_999)
        self.assertEqual(1, second - first)
        self.assertEqual(
            1,
            add_offset_nanoseconds(second, 1_737_987)
            - add_offset_nanoseconds(first, 1_737_987),
        )

    def test_clock_domain_residual_detects_boot_time(self):
        system_time_ns = stamp_to_nanoseconds(1_784_533_873, 0)
        boot_time_ns = stamp_to_nanoseconds(3_143, 0)
        residual_ns = clock_residual_nanoseconds(boot_time_ns, system_time_ns)
        self.assertLess(residual_ns, -1_000_000_000_000_000_000)

    def test_clock_domain_residual_preserves_submillisecond_sign(self):
        reference_ns = stamp_to_nanoseconds(100, 500_000)
        self.assertEqual(
            -200_000,
            clock_residual_nanoseconds(reference_ns - 200_000, reference_ns),
        )


class StrictStampGuardTest(unittest.TestCase):
    def test_rejects_duplicate_and_backward_without_advancing(self):
        guard = StrictStampGuard()
        self.assertIs(StampOrder.ACCEPT, guard.classify(100))
        guard.commit(100)
        self.assertIs(StampOrder.DUPLICATE, guard.classify(100))
        self.assertIs(StampOrder.NONMONOTONIC, guard.classify(99))
        self.assertEqual(100, guard.last_accepted_ns)
        self.assertIs(StampOrder.ACCEPT, guard.classify(101))
        guard.commit(101)
        self.assertEqual(101, guard.last_accepted_ns)

    def test_commit_refuses_non_increasing_stamp(self):
        guard = StrictStampGuard(100)
        with self.assertRaises(TimestampContractError):
            guard.commit(100)
        with self.assertRaises(TimestampContractError):
            guard.commit(99)


class ConsecutiveFailureGateTest(unittest.TestCase):
    def test_trips_on_third_consecutive_failure(self):
        gate = ConsecutiveFailureGate(3)
        self.assertFalse(gate.observe(True))
        self.assertFalse(gate.observe(True))
        self.assertTrue(gate.observe(True))
        self.assertEqual(3, gate.count)

    def test_healthy_observation_resets_the_streak(self):
        gate = ConsecutiveFailureGate(3)
        gate.observe(True)
        gate.observe(True)
        self.assertFalse(gate.observe(False))
        self.assertEqual(0, gate.count)
        self.assertFalse(gate.observe(True))


class MessageTransparencyTest(unittest.TestCase):
    def test_only_header_stamp_and_frame_are_changed(self):
        message = make_message()
        output = clone_with_aligned_stamp(message, "fcu_imu", 1_737_987)

        self.assertIsNot(message, output)
        self.assertEqual("base_link", message.header.frame_id)
        self.assertEqual((10, 20), (message.header.stamp.sec, message.header.stamp.nanosec))
        self.assertEqual("fcu_imu", output.header.frame_id)
        self.assertEqual(
            1_737_987,
            stamp_to_nanoseconds(
                output.header.stamp.sec,
                output.header.stamp.nanosec,
            ) - stamp_to_nanoseconds(10, 20),
        )
        self.assertEqual(message.orientation, output.orientation)
        self.assertEqual(message.angular_velocity, output.angular_velocity)
        self.assertEqual(message.linear_acceleration, output.linear_acceleration)
        self.assertEqual(
            message.angular_velocity_covariance,
            output.angular_velocity_covariance,
        )
        self.assertEqual(
            message.linear_acceleration_covariance,
            output.linear_acceleration_covariance,
        )
        for input_value, output_value in zip(
            message.orientation_covariance,
            output.orientation_covariance,
        ):
            if math.isnan(input_value):
                self.assertTrue(math.isnan(output_value))
            else:
                self.assertEqual(input_value, output_value)

    def test_constant_offset_preserves_sample_interval(self):
        first = make_message(sec=10, nanosec=100)
        second = make_message(sec=10, nanosec=200)
        first_output = clone_with_aligned_stamp(first, "fcu_imu", 1_737_987)
        second_output = clone_with_aligned_stamp(second, "fcu_imu", 1_737_987)
        input_interval = stamp_to_nanoseconds(10, 200) - stamp_to_nanoseconds(10, 100)
        output_interval = stamp_to_nanoseconds(
            second_output.header.stamp.sec,
            second_output.header.stamp.nanosec,
        ) - stamp_to_nanoseconds(
            first_output.header.stamp.sec,
            first_output.header.stamp.nanosec,
        )
        self.assertEqual(input_interval, output_interval)


if __name__ == "__main__":
    unittest.main()
