import torch
class MultiOctaveConv(torch.nn.Module):
    def safe_sum(self,a,b):
        if b is None:
            return a
        elif a is None:
            return b
        else:
            return a + b
        
    
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.25, alpha_out=0.25, 
                 beta_in=0.25,beta_out=0.25, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer = None, activation_layer = None, last=False):
        
        super(MultiOctaveConv, self).__init__()
        self.downsample_alpha = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.downsample_beta = torch.nn.AvgPool2d(kernel_size=(4, 4), stride=4)
        self.upsample_alpha = torch.nn.Upsample(scale_factor=2, mode='nearest')
        #self.upsample_beta = torch.nn.Upsample(scale_factor=4, mode='nearest')
        self.activation = activation_layer
        self.norm_layer =  norm_layer
        self.last = last
        self.even_k = (kernel_size %2 == 0 and not self.last)
        if norm_layer is not None:
            bias = False
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1 and 0 <= beta_in <= 1 and 0 <= beta_out <= 1, "Alphas should be in the interval from 0 to 1."
        assert 0 <= alpha_in + beta_in <= 1 and 0 <= alpha_out + beta_out <= 1, "Alphas_x + Beta_x should be in the interval from 0 to 1."
                
        if last:
            #print(self.even_k)
            kernel_size_l = kernel_size//2
            kernel_size_ll = kernel_size//4
        else:
            
            kernel_size_l = kernel_size
            kernel_size_ll = kernel_size
        
        alpha_in_channels = int(alpha_in * in_channels)
        beta_in_channels = int(beta_in * in_channels)
        gamma_in_channels = in_channels - alpha_in_channels -beta_in_channels
        
        self.alpha_out_channels = int(alpha_out * out_channels)
        self.beta_out_channels = int(beta_out * out_channels)
        self.gamma_out_channels = out_channels - self.alpha_out_channels -self.beta_out_channels
        #print(alpha_in_channels, beta_in_channels, gamma_in_channels,"=>",
        #      self.alpha_out_channels, self.beta_out_channels, self.gamma_out_channels)
        
        self.bn_h = None if self.gamma_out_channels == 0 or norm_layer is None else norm_layer(self.gamma_out_channels,  affine=True, track_running_stats=True)
        self.bn_l = None if self.alpha_out_channels == 0 or norm_layer is None else norm_layer(self.alpha_out_channels, affine=True, track_running_stats=True)
        self.bn_ll = None if self.beta_out_channels == 0 or norm_layer is None else norm_layer(self.beta_out_channels, affine=True, track_running_stats=True)
        if self.even_k :
            self.conv_h2l = None if gamma_in_channels == 0 or self.alpha_out_channels == 0 else \
                            torch.nn.Conv2d(gamma_in_channels, self.alpha_out_channels,
                                      kernel_size_l, 2, padding, dilation, groups, bias)
            self.conv_h2h = None if gamma_in_channels == 0 or self.gamma_out_channels == 0 else \
                            torch.nn.Conv2d(gamma_in_channels, self.gamma_out_channels,
                                      kernel_size, 2, padding, dilation, groups, bias)
            #self.conv_h2ll = None if gamma_in_channels == 0 or self.beta_out_channels == 0 else \
            #                torch.nn.Conv2d(gamma_in_channels, self.beta_out_channels,
            #                          kernel_size_ll, 1, padding, dilation, groups, bias)

            self.conv_l2l = None if alpha_in_channels == 0 or self.alpha_out_channels == 0 else \
                            torch.nn.Conv2d(alpha_in_channels, self.alpha_out_channels,
                                      kernel_size_l, 2, padding, dilation, groups, bias)
            self.conv_l2h = None if alpha_in_channels == 0 or self.gamma_out_channels == 0 else \
                            torch.nn.Conv2d(alpha_in_channels, self.gamma_out_channels,
                                      kernel_size_l, 2, padding, dilation, groups, bias)
            self.conv_l2ll = None if alpha_in_channels == 0 or self.beta_out_channels == 0 else \
                            torch.nn.Conv2d(alpha_in_channels, self.beta_out_channels,
                                      kernel_size_ll, 2, padding, dilation, groups, bias)

            self.conv_ll2l = None if beta_in_channels == 0 or self.alpha_out_channels == 0 else \
                            torch.nn.Conv2d(beta_in_channels, self.alpha_out_channels,
                                      kernel_size_ll, 2, padding, dilation, groups, bias)
            #self.conv_ll2h = None if beta_in_channels == 0 or self.gamma_out_channels == 0 else \
            #                torch.nn.Conv2d(beta_in_channels, self.gamma_out_channels,
            #                          kernel_size_ll, 1, padding, dilation, groups, bias)
            self.conv_ll2ll = None if beta_in_channels == 0 or self.beta_out_channels == 0 else \
                            torch.nn.Conv2d(beta_in_channels, self.beta_out_channels,
                                      kernel_size_ll, 2, padding, dilation, groups, bias)
        else:
            self.conv_h2l = None if gamma_in_channels == 0 or self.alpha_out_channels == 0 else \
                            torch.nn.Conv2d(gamma_in_channels, self.alpha_out_channels,
                                      kernel_size_l, 1, padding, dilation, groups, bias)
            self.conv_h2h = None if gamma_in_channels == 0 or self.gamma_out_channels == 0 else \
                            torch.nn.Conv2d(gamma_in_channels, self.gamma_out_channels,
                                      kernel_size, 1, padding, dilation, groups, bias)
            #self.conv_h2ll = None if gamma_in_channels == 0 or self.beta_out_channels == 0 else \
            #                torch.nn.Conv2d(gamma_in_channels, self.beta_out_channels,
            #                          kernel_size_ll, 1, padding, dilation, groups, bias)

            self.conv_l2l = None if alpha_in_channels == 0 or self.alpha_out_channels == 0 else \
                            torch.nn.Conv2d(alpha_in_channels, self.alpha_out_channels,
                                      kernel_size_l, 1, padding, dilation, groups, bias)
            self.conv_l2h = None if alpha_in_channels == 0 or self.gamma_out_channels == 0 else \
                            torch.nn.Conv2d(alpha_in_channels, self.gamma_out_channels,
                                      kernel_size_l, 1, padding, dilation, groups, bias)
            self.conv_l2ll = None if alpha_in_channels == 0 or self.beta_out_channels == 0 else \
                            torch.nn.Conv2d(alpha_in_channels, self.beta_out_channels,
                                      kernel_size_ll, 1, padding, dilation, groups, bias)

            self.conv_ll2l = None if beta_in_channels == 0 or self.alpha_out_channels == 0 else \
                            torch.nn.Conv2d(beta_in_channels, self.alpha_out_channels,
                                      kernel_size_ll, 1, padding, dilation, groups, bias)
            #self.conv_ll2h = None if beta_in_channels == 0 or self.gamma_out_channels == 0 else \
            #                torch.nn.Conv2d(beta_in_channels, self.gamma_out_channels,
            #                          kernel_size_ll, 1, padding, dilation, groups, bias)
            self.conv_ll2ll = None if beta_in_channels == 0 or self.beta_out_channels == 0 else \
                            torch.nn.Conv2d(beta_in_channels, self.beta_out_channels,
                                      kernel_size_ll, 1, padding, dilation, groups, bias)
    
    def forward(self, x):
        x_h, x_l, x_ll = x if type(x) is tuple else (x, None, None)
        out_x_h = None
        out_x_l = None
        out_x_ll = None
        if x_h is not None:
            #if self.last:
            #    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            x_h = self.downsample_alpha(x_h) if self.stride == 2 and not self.even_k else x_h
            #if self.last:
            #            print(x_h.shape)
            x_h2h = self.conv_h2h(x_h)  if self.gamma_out_channels >0 else None
            #if self.last:
            #            print(x_h2h.shape)
            x_h2l = self.conv_h2l(self.downsample_alpha(x_h)) if self.alpha_out_channels >0 else None
            #x_h2ll = self.conv_h2ll(self.downsample_beta(x_h)) if self.beta_out_channels >0 else None
            
            out_x_h = self.safe_sum(out_x_h,x_h2h)
            out_x_l = self.safe_sum(out_x_l,x_h2l)
            #out_x_ll = self.safe_sum(out_x_ll,x_h2ll)
        #print("pass h")
        if x_l is not None:
            x_l2h = self.conv_l2h(x_l) if self.gamma_out_channels >0 else None
            if x_l2h is not None:
                    #if self.last:
                    #print(x_l2h.shape)
                    x_l2h = self.upsample_alpha(x_l2h) if (self.stride == 1 or self.even_k) and not self.last else x_l2h
                    #if self.last:
                    #print(x_l2h.shape)
            x_l2l = self.downsample_alpha(x_l) if self.stride == 2 and not self.even_k else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out_channels > 0 else None 
            
            if self.stride ==2 and not self.even_k:
                x_l2ll = self.conv_l2ll(self.downsample_beta(x_l)) if self.beta_out_channels >0 else None
            else:    
                x_l2ll = self.conv_l2ll(self.downsample_alpha(x_l)) if self.beta_out_channels >0 else None
            
            out_x_h = self.safe_sum(out_x_h,x_l2h)
            out_x_l = self.safe_sum(out_x_l,x_l2l)
            out_x_ll = self.safe_sum(out_x_ll,x_l2ll)
        #print("pass l")
        
        
        if x_ll is not None:
            #x_ll2h = self.conv_ll2h(x_ll) if self.gamma_out_channels >0 else None
            #if x_ll2h is not None:
                #if self.last:
                #        print(x_ll2h.shape)
            #    if not self.last:
             #       x_ll2h = self.upsample_beta(x_ll2h) if self.stride == 1 else self.upsample_alpha(x_ll2h)
                #if self.last:
                #        print(x_ll2h.shape)
                        
            x_ll2l = self.conv_ll2l(x_ll) if self.alpha_out_channels >0 else None
            if x_ll2l is not None:
                x_ll2l = self.upsample_alpha(x_ll2l) if (self.stride == 1 or self.even_k) and not self.last  else x_ll2l
            x_ll2ll = self.downsample_alpha(x_ll) if self.stride == 2 and not self.even_k else x_ll
            x_ll2ll = self.conv_ll2ll(x_ll2ll) if self.beta_out_channels > 0 else None 
            
           # out_x_h = self.safe_sum(out_x_h,x_ll2h)
            out_x_l = self.safe_sum(out_x_l,x_ll2l)
            out_x_ll = self.safe_sum(out_x_ll,x_ll2ll)
        #print("pass ll")
        if self.norm_layer is not None:
            out_x_h = self.bn_h(out_x_h) if out_x_h is not None else None
            out_x_l = self.bn_l(out_x_l) if out_x_l is not None else None
            out_x_ll = self.bn_ll(out_x_ll) if out_x_ll is not None else None

        if self.activation is not None:
            out_x_h = self.activation(out_x_h) if out_x_h is not None else None
            out_x_l = self.activation(out_x_l) if out_x_l is not None else None
            out_x_ll =self.activation( out_x_ll) if out_x_ll is not None else None
        
        #if self.last:
        #        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        """
        if out_x_h is not None:
            if out_x_l is not None:
                if out_x_ll is not None:
                    print(out_x_h.shape, out_x_l.shape, out_x_ll.shape)
                else:
                    print(out_x_h.shape, out_x_l.shape)
            else:
                print(out_x_h.shape)
            
        """
        return out_x_h, out_x_l, out_x_ll


