# detection infe

def main_worker_detection(**kwargs):
    from engine.eval_detection import main_detection_evaluation
    main_detection_evaluation(**kwargs)


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main_worker_detection()
