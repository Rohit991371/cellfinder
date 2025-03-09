import click
import numpy as np
from cellfinder.core.detect import detect


@click.command()
@click.option('--input-file', required=True, type=click.Path(exists=True), help='Path to the input file containing the signal array.')
@click.option('--torch-device', default='cpu', help='Device to run the computation on (e.g., "cpu", "cuda").')
@click.option('--batch-size', default=None, type=int, help='Number of planes to process in each batch.')
@click.option('--use-scipy', is_flag=True, help='Use SciPy for certain computations.')
def main(input_file, torch_device, batch_size, use_scipy):
    """
    CLI entry point for cell detection.
    """
    # Load the signal array from the input file
    signal_array = np.load(input_file)

    start_plane = 0
    end_plane = -1
    voxel_sizes = (5, 2, 2)
    soma_diameter = 16
    max_cluster_size = 100_000
    ball_xy_size = 6
    ball_z_size = 15
    ball_overlap_fraction = 0.6
    soma_spread_factor = 1.4
    n_free_cpus = 2
    log_sigma_size = 0.2
    n_sds_above_mean_thresh = 10
    outlier_keep = False
    artifact_keep = False
    save_planes = False
    plane_directory = None
    split_ball_xy_size = 3
    split_ball_z_size = 3
    split_ball_overlap_fraction = 0.8
    split_soma_diameter = 7
    callback = None

    # Pass the options to the main detection function
    detect.main(
        signal_array=signal_array,
        start_plane=start_plane,
        end_plane=end_plane,
        voxel_sizes=voxel_sizes,
        soma_diameter=soma_diameter,
        max_cluster_size=max_cluster_size,
        ball_xy_size=ball_xy_size,
        ball_z_size=ball_z_size,
        ball_overlap_fraction=ball_overlap_fraction,
        soma_spread_factor=soma_spread_factor,
        n_free_cpus=n_free_cpus,
        log_sigma_size=log_sigma_size,
        n_sds_above_mean_thresh=n_sds_above_mean_thresh,
        outlier_keep=outlier_keep,
        artifact_keep=artifact_keep,
        save_planes=save_planes,
        plane_directory=plane_directory,
        batch_size=batch_size,
        torch_device=torch_device,
        use_scipy=use_scipy,
        split_ball_xy_size=split_ball_xy_size,
        split_ball_z_size=split_ball_z_size,
        split_ball_overlap_fraction=split_ball_overlap_fraction,
        split_soma_diameter=split_soma_diameter,
        callback=callback,
    )


if __name__ == "__main__":
    main()
