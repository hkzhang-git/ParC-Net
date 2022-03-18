# segmentation infe

def main_worker_segmentation(**kwargs):
    from engine.eval_segmentation import main_segmentation_evaluation
    main_segmentation_evaluation(**kwargs)


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main_worker_segmentation()

