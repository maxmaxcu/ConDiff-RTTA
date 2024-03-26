import torch
import torch.nn as nn

class TTABase_Class(nn.Module):
    """Define a general interface for TTA, which is applicable for discrete
    sampling and gradient-descent based TTA.

    Args:
        unet_model: A nn.Module indicates the U-Net in the diffusion model
        class_model: A nn.Module indicates the classifier
    """
    def __init__(self, config, unet_model, class_model, batch_size):
        super().__init__()
        self.unet_model = unet_model
        self.class_model = class_model
        self.before_tta_preds = []
        self.after_tta_preds = []
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data=0.5
        self.batch_size = batch_size
        self.config = config
        if config.tta.model.classifier_init_method != 'rand':
            self.noneclass_model = nn.Parameter(torch.ones((batch_size,1))*config.tta.model.classifier_init_method)
            if not config.tta.model.use_pretrained_classifier:
                print(f'classifier_init_method:{config.tta.model.classifier_init_method}')
        else:
            self.noneclass_model = nn.Parameter(torch.rand((batch_size,1)))
            if not config.tta.model.use_pretrained_classifier:
                print(f'classifier_init_method:{config.tta.model.classifier_init_method}')
        
        self.dv_initvalue = nn.Parameter(torch.zeros((batch_size, 31)))
        
        self.image_diff_to_trans = nn.Parameter(torch.zeros((1, 1, 64, 64)))
          
    def evaluate(self, images_diff, gt, before_tta=False, after_tta=False):
        """Implement this function in subclasses.
        """
        raise NotImplementedError

    def _unet_pred_noise(self, x_start, context, bs, config):
        x_start = x_start.expand(bs, -1, -1, -1)
        context = context.expand(bs, -1,)
        rnd_normal = torch.randn([x_start.shape[0], 1, 1, 1], device=x_start.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = x_start, None
        n = torch.randn_like(y) * sigma
        D_yn = self.unet_model(y + n, sigma, context, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        loss = loss.mean()
        return loss
    
  
    def classify(self, image):
        """A helper function to outputs classification results

        Args:
            image: A tensor of shape [1, 3, H, W]

        """
        # Classify with the classifier
        if self.config.tta.model.use_pretrained_classifier == True:
            preds = self.class_model(image)
        elif self.config.tta.model.use_pretrained_classifier == 'ValueOnly':
            preds = self.class_model(image)
        else:
            preds = self.noneclass_model
        return preds


    def forward(self, image_discr, image_diff):
        """Perform classification or compute diffusion loss.

        Args:
            image: A tensor of shape [1, 3, H, W]
            x_start: A tensor of shape [1, C, latent_H, latent_W]
            t: A tensor of shape [num_timesteps]
            noise: A tensor of shape [num_timesteps, C, latent_H, latent_W]
            pred_top_idx: A tensor of shape [1, K]
        """
        raise NotImplementedError
    


class TTAGradientDescent_Class(TTABase_Class):
    
            
    def evaluate(self, images_diff, gt, before_tta=False, after_tta=False, bs=None, config=None):
        """Evaluate classifier predictions
        """
        # Classify with the classifier
        if self.config.tta.model.use_pretrained_classifier == True:
            
            with torch.no_grad():
                self.class_model.eval()
                image = images_diff
                preds = self.classify(image)
                preds = (preds)
        else:
            preds = (self.noneclass_model)


        # Keep track of the correctness among all images
        if before_tta:
            with torch.no_grad():
                prefix = 'before_tta'
                self.before_tta_preds.append(preds)
        elif after_tta:
            self.after_tta_preds.append(preds)
            prefix = 'after_tta'
        else:
            prefix = ''
        stats_dict = {}
        stats_dict[f'{prefix}_preds'] = preds
        stats_dict[f'{prefix}_gt'] = gt

        return stats_dict
    

    def forward(self, image_discr, image_diff, bs, config):
        """This function compute diffusion loss using current classifier
        predictions.
        """
        # Classify with the classifier
        if self.config.tta.model.use_pretrained_classifier == True:
            preds = self.classify(image_discr)
        else:
            preds = self.noneclass_model
        loss = self._unet_pred_noise(image_diff, preds, bs, config)
        return loss
    

  
        