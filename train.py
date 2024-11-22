
import cv2
import numpy as np
import argparse
import gc
import math
import os
import toml
from multiprocessing import Value
from PIL import Image

from tqdm import tqdm
from accelerate.utils import set_seed
import diffusers
from diffusers import DDPMScheduler

import library.train_util as train_util
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import apply_snr_weight

from torchvision import models
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import torchvision.transforms as T


class VGGFeatureExtractor(nn.Module):

    def __init__(self, feature_layers):
        super(VGGFeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features

        self.features = nn.ModuleList()
        self.feature_layers = feature_layers

        current_layer = 0
        for layer in vgg19:
            if isinstance(layer, nn.Conv2d):
                current_layer += 1
            self.features.append(layer)
            if current_layer > max(feature_layers):
                break

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        current_layer = 0

        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                current_layer += 1
                if current_layer in self.feature_layers:
                    features.append(x)
        return features

class StyleContentLoss():

    def __init__(self, content_layers=[4], style_layers=[1, 6, 11, 20],
                 content_weight=0.01, style_weight=1.0, device='cuda'):
        self.content_extractor = VGGFeatureExtractor(content_layers).to(device)
        self.style_extractor = VGGFeatureExtractor(style_layers).to(device)
        self.content_weight = content_weight
        self.style_weight = style_weight

    def preprocess_images(self, generated_images, target_images):

        if generated_images.min() < 0:
            generated_images = (generated_images + 1.0) / 2.0
        generated_images = torch.clamp(generated_images, min=0.0, max=1.0)

        if target_images.max() > 1.0:
            target_images = target_images / 255.0
        target_images = torch.clamp(target_images, min=0.0, max=1.0)

        return generated_images, target_images


    def compute_content_loss(self, generated_images, target_images):

        content_features_gen = self.content_extractor(generated_images)
        content_features_target = self.content_extractor(target_images)

        content_loss = F.mse_loss(content_features_gen[0], content_features_target[0])

        content_loss = content_loss / (content_features_gen[0].std() + 1e-8)

        return content_loss

    def compute_histogram_loss(self, generated_images, target_image, num_bins=256):

        gen_hist = torch.histc(generated_images.view(-1), bins=num_bins, min=0, max=1)
        target_hist = torch.histc(target_image.view(-1), bins=num_bins, min=0, max=1)
        gen_hist /= gen_hist.sum() + 1e-8
        target_hist /= target_hist.sum() + 1e-8
        return F.mse_loss(gen_hist, target_hist)

    def compute_style_loss(self, generated_images, target_images):
        style_loss = 0.0
        num_batches = len(generated_images)

        for gen_image, target_image in zip(generated_images, target_images):
            style_loss += self.compute_histogram_loss(gen_image, target_image)

        style_loss = style_loss / (num_batches + 1e-8)

        return style_loss

    def compute_loss(self, generated_images, target_images, writer=None, global_step=None):

        generated_images, target_images = self.preprocess_images(generated_images, target_images)

        generated_images = generated_images.to(torch.float32)
        target_images = target_images.to(torch.float32)

        content_loss = self.compute_content_loss(generated_images, target_images)

        style_loss = self.compute_style_loss(generated_images, target_images)

        total_loss = (self.content_weight * content_loss +
                      self.style_weight * style_loss)

        if writer is not None and global_step is not None:
            writer.add_scalar('content_loss', content_loss.item(), global_step)
            writer.add_scalar('style_loss', style_loss.item(), global_step)
            writer.add_scalar('total_loss', total_loss.item(), global_step)

        return total_loss, content_loss, style_loss


class EdgeFeatureExtractor(nn.Module):
    def __init__(self):
        super(EdgeFeatureExtractor, self).__init__()

        self.backbone = models.resnet18(pretrained=True)

        self.backbone.fc = nn.Identity()


        self.mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, edge_tensor: torch.Tensor):

        features = self.backbone(edge_tensor)

        features_1024 = self.mlp(features)

        sequence_length = 1
        feature_dim = 1024

        reshaped_features = features_1024.view(edge_tensor.size(0), sequence_length, feature_dim)

        return reshaped_features

def preprocess_images(edge_images):

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    edge_tensors = []
    for img in edge_images:
        tensor = transform(img)
        edge_tensors.append(tensor)

    edge_tensors = torch.stack(edge_tensors, dim=0)
    return edge_tensors


def train(args):

    train_util.verify_training_args(args)

    train_util.prepare_dataset_args(args, True)

    cache_latents = args.cache_latents

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = train_util.load_tokenizer(args)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer(False, True, True))

    if args.dataset_config is not None:
        print(f"Load dataset config from {args.dataset_config}")
        user_config = config_util.load_user_config(args.dataset_config)
        ignored = ["train_data_dir", "in_json"]
        if any(getattr(args, attr) is not None for attr in ignored):
            print(
                "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                    ", ".join(ignored)
                )
            )

    else:
        user_config = {
            "datasets": [
                {
                    "subsets": [
                        {
                            "image_dir": args.train_data_dir,
                            "metadata_file": args.in_json,
                        }
                    ]
                }
            ]
        }

    blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)


    train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    print(train_dataset_group)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)

    ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None

    collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group)
        return


    if len(train_dataset_group) == 0:
        print(
            "No data found. Please verify the metadata file and train_data_dir option. "
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used "

    print("prepare accelerator")
    accelerator, unwrap_model = train_util.prepare_accelerator(args)

    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    text_encoder, vae, unet, load_stable_diffusion_format = train_util.load_target_model(args, weight_dtype)

    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path


    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:

        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())


    def set_diffusers_xformers_flag(model, valid):

        def fn_recursive_set_mem_eff(module: torch.nn.Module):

            if hasattr(module, "set_use_memory_efficient_attention_xformers"):

                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    if args.diffusers_xformers:
        print("Use xformers by Diffusers")
        set_diffusers_xformers_flag(unet, True)

    else:
        print("Disable Diffusers' xformers")
        set_diffusers_xformers_flag(unet, False)
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers)

    if cache_latents:

        vae.to(accelerator.device, dtype=weight_dtype)

        vae.requires_grad_(False)
        vae.eval()

        with torch.no_grad():
            train_dataset_group.cache_latents(vae, args.vae_batch_size)
        vae.to("cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
    training_models = []

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    training_models.append(unet)

    if args.train_text_encoder:
        print("enable text encoder training")

        if args.gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()

        training_models.append(text_encoder)
    else:

        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)

        if args.gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
            text_encoder.train()
        else:

            text_encoder.eval()

    if not cache_latents:

        vae.requires_grad_(False)
        vae.eval()

        vae.to(accelerator.device, dtype=weight_dtype)

    for m in training_models:
        m.requires_grad_(True)

    params = []
    for m in training_models:
        params.extend(m.parameters())
    params_to_optimize = params


    print("prepare optimizer, data loader etc.")

    _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)


    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collater,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        print(f"override steps. steps for {args.max_train_epochs} epochs is : {args.max_train_steps}")

    train_dataset_group.set_max_train_steps(args.max_train_steps)

    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder.to(weight_dtype)

    if args.train_text_encoder:

        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:

        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):

        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1


    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("Starting training")
    print(f"  Number of samples: {train_dataset_group.num_train_images}")
    print(f"  Number of batches per epoch: {len(train_dataloader)}")
    print(f"  Number of epochs: {num_train_epochs}")
    print(f"  Batch size per device: {args.train_batch_size}")
    print(f"  Total training batch size (including parallel, distributed, and accumulated): {total_batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps),
                        smoothing=0,
                        disable=not accelerator.is_local_main_process,
                        desc="steps")
    global_step = 0


    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("finetuning")

    loss_calculator = StyleContentLoss(
        content_layers=[4],
        style_layers=[1, 6, 11, 20],
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        device=accelerator.device
    )

    resnet_backbone = models.resnet18(pretrained=True)

    device = torch.device("cuda")

    writer = SummaryWriter(log_dir='logs')

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_train_epochs):
        print(f"epoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        loss_total = 0


        for step, batch in enumerate(train_dataloader):

            current_step.value = global_step

            with accelerator.accumulate(training_models[0]):

                with torch.no_grad():
                    if "latents" in batch and batch["latents"] is not None:

                        latents = batch["latents"].to(accelerator.device).to(torch.float32)
                    else:

                        latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()

                    latents = latents * 0.18215

                b_size = latents.shape[0]


                edge_images = [Image.open(path).convert("RGB") for path in batch["edge_paths"]]

                edge_tensors = preprocess_images(edge_images)


                feature_extractor = EdgeFeatureExtractor().cuda()


                edge_features = feature_extractor(edge_tensors.cuda())

                with torch.set_grad_enabled(args.train_text_encoder):

                    input_ids = batch["input_ids"].to(accelerator.device)

                    encoder_hidden_states = train_util.get_hidden_states(
                        args, input_ids, tokenizer, text_encoder, None if not args.full_fp16 else weight_dtype
                    )


                concatenated_features = torch.cat((encoder_hidden_states, edge_features), dim=1)

                noise = torch.randn_like(latents, device=latents.device)

                if args.noise_offset:
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                noise_pred = unet(noisy_latents, timesteps, concatenated_features).sample

                if args.v_parameterization:
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)

                else:
                    target = noise

                if args.min_snr_gamma:

                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean([1, 2, 3])
                    loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)
                    loss = loss.mean()
                else:
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                writer.add_scalar("loss", loss, global_step)

                previous_samples = []

                for t, pred, noisy in zip(timesteps, noise_pred, noisy_latents):
                    int_t = t.item()
                    current_latent = noise_scheduler.step(pred, int_t, noisy).prev_sample
                    previous_samples.append(current_latent)


                previous_samples_tensor = torch.stack(previous_samples).to(dtype=torch.float16, device='cuda')

                # 解码生成图像
                generated_images = vae.decode(previous_samples_tensor).sample

                result = torch.clamp((generated_images + 1.0) / 2.0, min=0.0, max=1.0)
                result = result.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255

                for i in range(result.shape[0]):

                    img = result[i].astype(np.uint8)

                    image = Image.fromarray(img)

                    image.show(title=f"Generated Image {i + 1}")

                    image.save(f"generated_image_{i + 1}.png")

                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) / 255.0

                    writer.add_image(f'Generated Image {i + 1}', img_tensor.squeeze(0), global_step=step)

                images_B = [Image.open(path).convert("RGB") for path in batch["domainB_paths"]]

                image_tensors = [torch.tensor(np.array(img)).permute(2, 0, 1) for img in images_B]

                images_B_tensor = torch.stack(image_tensors)

                images_B_tensor = images_B_tensor / 255.0

                images_B_tensor = images_B_tensor.to(accelerator.device)

                generated_images = generated_images.to(torch.float16)
                images_B_tensor = images_B_tensor.to(torch.float16)

                total_loss, content_loss, style_loss = loss_calculator.compute_loss(
                    generated_images,
                    images_B_tensor,
                    writer=writer,
                    global_step=global_step
                )

                loss = loss + total_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = []
                    for m in training_models:
                        params_to_clip.extend(m.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                train_util.sample_images(
                    accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet
                )

            current_loss = loss.detach().item()
            if args.logging_dir is not None:
                logs = {"loss": current_loss, "lr": float(lr_scheduler.get_last_lr()[0])}
                if args.optimizer_type.lower() == "DAdaptation".lower():
                    logs["lr/d*lr"] = (
                        lr_scheduler.optimizers[0].param_groups[0]["d"] * lr_scheduler.optimizers[0].param_groups[0]["lr"]
                    )

                writer.add_scalar("Loss/train", current_loss, global_step)
                writer.add_scalar("Learning_rate/train", logs["lr"], global_step)
                accelerator.log(logs, step=global_step)

            # TODO moving averageにする
            loss_total += current_loss
            avr_loss = loss_total / (step + 1)
            logs = {"loss": avr_loss}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_total / len(train_dataloader)}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
            train_util.save_sd_model_on_epoch_end(
                args,
                accelerator,
                src_path,
                save_stable_diffusion_format,
                use_safetensors,
                save_dtype,
                epoch,
                num_train_epochs,
                global_step,
                unwrap_model(text_encoder),
                unwrap_model(unet),
                vae,
            )

        train_util.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

    is_main_process = accelerator.is_main_process

    if is_main_process:

        unet = unwrap_model(unet)
        text_encoder = unwrap_model(text_encoder)

    accelerator.end_training()

    if args.save_state:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator

    if is_main_process:

        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path

        train_util.save_sd_model_on_train_end(
            args,
            src_path,
            save_stable_diffusion_format,
            use_safetensors,
            save_dtype,
            epoch,
            global_step,
            text_encoder,
            unet,
            vae
        )
        print("model saved.")

    writer.close()

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)

    train_util.add_dataset_arguments(parser, False, True, True)

    train_util.add_training_arguments(parser, False)

    train_util.add_sd_saving_arguments(parser)

    train_util.add_optimizer_arguments(parser)

    config_util.add_config_arguments(parser)

    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument("--diffusers_xformers", action="store_true", help="Use xformers in diffusers")

    parser.add_argument("--train_text_encoder", action="store_true", help="Train the text encoder")

    parser.add_argument("--content_weight", type=float, default=1.0, help="Weight for content loss")

    parser.add_argument("--style_weight", type=float, default=0.5, help="Weight for style loss")

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    args = train_util.read_config_from_file(args, parser)

    train(args)