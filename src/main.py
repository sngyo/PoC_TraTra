from data_utils.data_setup import generate_exclude_filenamelist, resampling_unp
from data_utils.iamonline import convert_xml_to_unp, gen_iam_single_stroke_data
from scripts.test import run_test
from scripts.train import train_tratra
from utils import log_init, utils
from utils.arguments import get_configs, save_configs


def main(configs):
    # -------------------------------------------------------------------------
    # scripts
    # -------------------------------------------------------------------------
    if configs.train_tratra:
        train_tratra(configs)
    elif configs.test_tratra:
        run_test(configs)

    # -------------------------------------------------------------------------
    elif configs.cvt_xml2unp:
        convert_xml_to_unp(configs.iam_xml_path, configs.iam_unp_path)
    elif configs.gen_iam_single_stroke_data:
        gen_iam_single_stroke_data(
            configs.iam_unp_path,
            configs.gen_img_h,
            configs.gen_img_w,
            configs.gen_img_pad,
            configs.gen_iam_path,
        )
    elif configs.gen_iam_resampled_data:
        resampling_unp(
            configs.gen_iam_path,
            configs.gen_iam_path,
            alpha=configs.rs_alpha,
            resample_dist=configs.rs_dist,
            max_p=configs.rs_max_p,
        )
    elif configs.gen_iam_exclude_filename_data:
        generate_exclude_filenamelist(
            configs.gen_iam_path,
            configs.rs_max_p,
            configs.rs_dist,
        )
    else:
        logger.warning("You should specify one script.")


if __name__ == "__main__":
    # logger
    logger = log_init.logger
    logger.info("script start!")

    # get configurations
    configs = get_configs(logger)
    save_configs(configs, logger)

    # seed fix
    utils.seed_everything(configs.seed)

    # run script
    main(configs)
    logger.info("Successfully finished!")
