from data_load import load_data
from model import define_model
#from harmonize import generate_harmonized_image
from model import define_diffusion
#from training import train_model
from training_perceptual_loss import train_model
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = load_data()
    model = define_model().to(device)
    #model = define_model()
    diffusion = define_diffusion(model)
    train_model(dataloader, model, diffusion)


if __name__ == "__main__":
    main()


"""
def main():
    source_image_path = "/home/youssef/harmonization_project/data/train_2/Pat8_CHU_zscore_minmax_unbias.nii.gz"
    target_image_path = "/home/youssef/harmonization_project/data/train_2/Pat8_COL_zscore_minmax_unbias.nii.gz"

    dataloader = load_data(source_image_path, target_image_path)
    model = define_model()
    diffusion = define_diffusion(model)
    train_model(dataloader, model, diffusion)

    # Generate and save a harmonized image using the trained model
    for source_slice, target_slice in dataloader:
        generate_harmonized_image('savedmodel.pt', source_slice, target_slice)
    

if __name__ == "__main__":
    main()
"""
