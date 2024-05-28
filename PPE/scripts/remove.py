from glob import glob
# from tqdm import tqdm


def main():
    #root_dir = "./Testcase/"
    #files = glob(f"{root_dir}/csv_*.csv")
    files = glob(f"./data_2d/out_*.csv")
    for file_ in (files):
        with open(file_, "r+") as f:
            txt = f.read().replace(" ", "")
            f.seek(0)
            f.write(txt)
            f.truncate()

    for file_ in (files):
        with open(file_, "r+") as f:
            txt = f.read().replace("E", "e")
            f.seek(0)
            f.write(txt)
            f.truncate()


if __name__ == "__main__":
    main()