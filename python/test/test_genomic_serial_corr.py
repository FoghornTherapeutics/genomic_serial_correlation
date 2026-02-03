import unittest
import logging
import fhtbioinfpy.setup_logger as setup_logger
import fhtbioinfpy.genomic_serial_corr.genomic_serial_corr as gensc

import numpy


logger = logging.getLogger(setup_logger.LOGGER_NAME)

# Some notes on testing conventions (more in cuppers convention doc):
#    (1) Use "self.assert..." over "assert"
#        - self.assert* methods: https://docs.python.org/2.7/library/unittest.html#assert-methods
#       - This will ensure that if one assertion fails inside a test method,
#         exectution won't halt and the rest of the test method will be executed
#         and other assertions are also verified in the same run.
#     (2) For testing exceptions use:
#        with self.assertRaises(some_exception) as context:
#            [call method that should raise some_exception]
#        self.assertEqual(str(context.exception), "expected exception message")
#
#        self.assertAlmostEquals(...) for comparing floats


class TestGenomicSerialCorr(unittest.TestCase):
    def test_main_functional(self):
        """ usually when testing main we have to do that as a functional test - we pick a reasonable set of parameters for a 
        happy path and run through those to make sure that works
        """
        gensc.main(None)

    def test_calculate_pad_to_length(self):
        max_array_length = 1000
        pad_to_length = gensc.calculate_pad_to_length(max_array_length)
        logger.debug("pad_to_length:  {}".format(pad_to_length))
        self.assertEqual(pad_to_length, 1024)

        max_array_length = 2048
        pad_to_length = gensc.calculate_pad_to_length(max_array_length)
        logger.debug("pad_to_length:  {}".format(pad_to_length))
        self.assertEqual(pad_to_length, 2048)

        max_array_length = 2049
        pad_to_length = gensc.calculate_pad_to_length(max_array_length)
        logger.debug("pad_to_length:  {}".format(pad_to_length))
        self.assertEqual(pad_to_length, 4096)

        
    def test_calculate_normalized_coverage(self):
        coverage_arr = numpy.array([0, 1, 2, 3, 4, 5])
        logger.debug("coverage_arr:  {}".format(coverage_arr))

        normalized_coverage = gensc.calculate_normalized_coverage(coverage_arr)
        logger.debug("normalized_coverage:  {}".format(normalized_coverage))

        expected_mean = 0.0
        expected_std = 1.0

        actual_mean = numpy.mean(normalized_coverage)
        actual_std = numpy.std(normalized_coverage)

        logger.debug("actual_mean:  {}".format(actual_mean))
        logger.debug("actual_std:  {}".format(actual_std))

        self.assertAlmostEqual(actual_mean, expected_mean)
        self.assertAlmostEqual(actual_std, expected_std)

    def test_calculate_rfft(self):
        coverage_arr = numpy.zeros(1023)

        coverage_arr[:] = numpy.sin(numpy.linspace(0, 4*numpy.pi, 1023))  # sinusoidal coverage
        # coverage_arr[::3] = 1  # every third position has coverage of 1, rest are zero

        r = gensc.calculate_rfft(coverage_arr, pad_to_length=1024)
        logger.debug("r.shape:  {}".format(r.shape))
        logger.debug("r[:10]:\n{}".format(r[:10]))

        self.assertEqual(r.shape[0], 513)

        power_spectrum = r * numpy.conj(r)
        logger.debug("power_spectrum[:10]:\n{}".format(power_spectrum[:10]))

        sum_power_spec_imag = numpy.sum(numpy.abs(numpy.imag(power_spectrum)))
        logger.debug("sum_power_spec_imag:  {}".format(sum_power_spec_imag))
        self.assertLess(sum_power_spec_imag, 1e-10)

        power_spectrum = numpy.real(power_spectrum)
        expected_freq_power_spec_val = power_spectrum[2]
        logger.debug("expected_freq_power_spec_val:  {}".format(expected_freq_power_spec_val))
        self.assertGreater(expected_freq_power_spec_val, 250_000.)


    def test_calculate_cross_correlation(self):
        my_rfft_1 = numpy.zeros(513, dtype=complex)
        my_rfft_1[2] = 500 + 0j  # frequency component at index 2

        my_rfft_2 = numpy.zeros(513, dtype=complex)
        my_rfft_2[2] = 200 + 0j  # frequency

        cross_corr = gensc.calculate_cross_correlation(my_rfft_1, my_rfft_2)
        logger.debug("cross_corr.dtype:  {}".format(cross_corr.dtype))
        logger.debug("cross_corr.shape:  {}".format(cross_corr.shape))
        logger.debug("cross_corr[:10]:\n{}".format(cross_corr[:10]))

        self.assertEqual(cross_corr.shape[0], 1024)

        t = numpy.argsort(cross_corr)
        logger.debug("t:  {}".format(t))

        minima_indexes = set(t[:2])
        maxima_indexes = set(t[-2:])
        logger.debug("minima_indexes:  {}".format(minima_indexes))
        logger.debug("maxima_indexes:  {}".format(maxima_indexes))

        self.assertEqual(minima_indexes, {256, 768})
        self.assertEqual(maxima_indexes, {0, 512})


    def test_build_centered_cross_correlation_array(self):
        logger.debug("test odd length array width None")
        my_cross_corr = numpy.array([0,1,2,3,5,7,13])
        logger.debug("my_cross_corr.shape:  {}".format(my_cross_corr.shape))
        logger.debug("my_cross_corr:  {}".format(my_cross_corr))
        r = gensc.build_centered_cross_correlation_array(my_cross_corr)
        logger.debug("r.shape:  {}".format(r.shape))
        logger.debug("r:  {}".format(r))
        expected = [5,7,13,0,1,2,3]
        comparison = r == expected
        self.assertTrue(all(comparison)), "expected:  {} detailed comparison:  {}".format(expected, comparison)

        logger.debug("test even length array width None")
        my_cross_corr = numpy.array([0,1,2,3,5,7])
        logger.debug("my_cross_corr.shape:  {}".format(my_cross_corr.shape))
        logger.debug("my_cross_corr:  {}".format(my_cross_corr))
        r = gensc.build_centered_cross_correlation_array(my_cross_corr)
        logger.debug("r.shape:  {}".format(r.shape))
        logger.debug("r:  {}".format(r))
        expected = [3,5,7,0,1,2]
        comparison = r == expected
        self.assertTrue(all(comparison)), "expected:  {} detailed comparison:  {}".format(expected, comparison)

        logger.debug("test odd length array width not None")
        my_cross_corr = numpy.array(range(101))
        logger.debug("my_cross_corr.shape:  {}".format(my_cross_corr.shape))
        logger.debug("my_cross_corr:  {}".format(my_cross_corr))
        r = gensc.build_centered_cross_correlation_array(my_cross_corr, 10)
        logger.debug("r.shape:  {}".format(r.shape))
        logger.debug("r:  {}".format(r))
        expected = list(range(91,101)) + list(range(0,10))
        comparison = r == expected
        self.assertTrue(all(comparison)), "expected:  {} detailed comparison:  {}".format(expected, comparison)

        logger.debug("test even length array width not None")
        my_cross_corr = numpy.array(range(100))
        logger.debug("my_cross_corr.shape:  {}".format(my_cross_corr.shape))
        logger.debug("my_cross_corr:  {}".format(my_cross_corr))
        r = gensc.build_centered_cross_correlation_array(my_cross_corr, 10)
        logger.debug("r.shape:  {}".format(r.shape))
        logger.debug("r:  {}".format(r))
        expected = list(range(90,100)) + list(range(0,10))
        comparison = r == expected
        self.assertTrue(all(comparison)), "expected:  {} detailed comparison:  {}".format(expected, comparison)

        logger.debug("")
        logger.debug("test raises exception when width too large")
        logger.debug("gensc.gensc.FhtbioinfpyGenomicSerialCorrBuildCenteredCrossCorrelationArrayWidthTooLargeException:  {}".format(gensc.FhtbioinfpyGenomicSerialCorrBuildCenteredCrossCorrelationArrayWidthTooLargeException))
        with self.assertRaises(gensc.FhtbioinfpyGenomicSerialCorrBuildCenteredCrossCorrelationArrayWidthTooLargeException) as context:
            gensc.build_centered_cross_correlation_array(my_cross_corr, 10*my_cross_corr.shape[0])
        logger.debug("context.exception:  {}".format(context.exception))
        self.assertIn("build_centered_cross_correlation_array width must be less than helf of cross_correlation array size", str(context.exception))


if __name__ == "__main__":
    setup_logger.setup(verbose=True)

    unittest.main()
