import logging
import fhtbioinfpy
import fhtbioinfpy.setup_logger as setup_logger
import argparse
import sys

import numpy
import scipy.fft as fft


logger = logging.getLogger(setup_logger.LOGGER_NAME)


DEFAULT_PAD_TO_LENGTH = 2**28


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", "-v", help="Whether to print a bunch of output.", action="store_true", default=False)
    # parser.add_argument("--hostname", help="lims db host name", type=str, default="getafix-v")

    # parser.add_argument("--config_filepath", help="path to config file containing information about how to connect to CDD API, ArxLab API etc.",
    #     type=str, default=fhtbioinfpy.default_config_filepath)
    # parser.add_argument("--config_section", help="section of config file to use for information about how to connect to CDD API, ArxLab API etc.",
        # type=str, default=fhtbioinfpy.default_config_section)

    # parser.add_argument("--queue_choice", "-qc", help="which of the queues to work on - valid values are roast, brew, both", type=str,
    #     choices=["roast", "brew", "both"], default="both")
    # parser.add_argument("--add_to_queue", "-a", help="add the det_plate entries to the roast_queue", type=str, nargs="+", default=None)

    # To make two options mutually exclusive, one can define mutually_exclusive_group in argparse,
    # argparse asserts that the options added to the group are not used at the same time and throws exception if otherwise
    # mutually_exclusive_group = parser.add_mutually_exclusive_group()
    # mutually_exclusive_group.add_argument("--analysis_dir", "-ad", type=str, 
    #     help="path to the analysis directory where DGE files can be found and output directory created")
    # mutually_exclusive_group.add_argument("--other_dir", "-od", type=str, 
    #     help="some other directory")
    
    # NB nesting an argument group within a mutually exclusive group causes the nested group not be displayed 
    # with the -h option (it is displayed if no command line arguments are given)
    # input_output_dirs_group = mutually_exclusive_group.add_argument_group("this nested group will not be displayed with -h")
    # input_output_dirs_group.add_argument("--dge_dir", "-dd", type=str,
    #     help="path to the input DGE directory containing standard gct files of logFC, adjPVal etc.")
    # input_output_dirs_group.add_argument("--output_dir", "-od", type=str, help="path to the output directory", required=True)

    return parser


def calculate_pad_to_length(max_array_length):
    """ calculate the pad to length for FFT calculation
    """
    power_of_2 = int(numpy.ceil(numpy.log2(max_array_length)))
    logger.debug("max_array_length:  {}  power_of_2:  {}".format(max_array_length, power_of_2))
    pad_to_length = 2**power_of_2
    return pad_to_length


def calculate_normalized_coverage(coverage_arr):
    """ calculate the normalized coverage of the input array for subsequent serial correlation calculation
    """
    mean_coverage = numpy.mean(coverage_arr)
    std_coverage = numpy.std(coverage_arr)
    normalized_coverage = (coverage_arr - mean_coverage) / std_coverage
    return normalized_coverage


def calculate_rfft(coverage_arr, pad_to_length=DEFAULT_PAD_TO_LENGTH):
    """ calculate the real FFT of the input array and return it
    """
    new_arr = numpy.zeros(pad_to_length)
    new_arr[: len(coverage_arr)] = coverage_arr

    r = fft.rfft(new_arr, overwrite_x=True)
    return r


def calculate_cross_correlation(rfft_1, rfft_2):
    """ calculate the cross correlation of two input arrays given their real FFTs
    """
    cross_power_spectrum = rfft_1 * numpy.conj(rfft_2)
    cross_correlation = fft.irfft(cross_power_spectrum, overwrite_x=True)
    return cross_correlation


def build_centered_cross_correlation_array(cross_correlation, width=None):
    """build_centered_cross_correlation_array
    take the output of calculate_cross_correlation and rearrange into an array that has data points
    ordered by lag

    width if width is None use the entire array, otherwise width is the target width on either side of lag=0.  Width must be <= half the size of cross_correlation
    """ 

    if width is None:
        half_shape = cross_correlation.shape[0] // 2
        left_limit = half_shape if cross_correlation.shape[0]%2 == 0 else half_shape+1
        right_limit = half_shape

    else:
        if 2*width > cross_correlation.shape[0]:
            raise FhtbioinfpyGenomicSerialCorrBuildCenteredCrossCorrelationArrayWidthTooLargeException("fhtbioinfpy genomic_serial_corr build_centered_cross_correlation_array width must be less than helf of cross_correlation array size")

        left_limit = width
        right_limit = width

    logger.debug("width={}, left_limit:  {}  right_limit:  {}".format(width, left_limit, right_limit))

    result = numpy.hstack([cross_correlation[-right_limit:],cross_correlation[:left_limit]])
    return result


class FhtbioinfpyGenomicSerialCorrBuildCenteredCrossCorrelationArrayWidthTooLargeException(Exception):
    pass


def main(args):
    logger.info("hello world!")


if __name__ == "__main__":
    args = build_parser().parse_args(sys.argv[1:])

    setup_logger.setup(verbose=args.verbose)

    logger.debug("args:  {}".format(args))

    main(args)
