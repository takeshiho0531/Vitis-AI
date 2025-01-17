# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class SSD(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(SSD, self).__init__()
        self.module_0 = py_nndct.nn.Module('nndct_const') #SSD::4538(SSD::nndct_const_0)
        self.module_1 = py_nndct.nn.Input() #SSD::input_0(SSD::nndct_input_1)
        self.module_2 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[0]/ret.3(SSD::nndct_conv2d_2)
        self.module_3 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[1]/3297(SSD::nndct_relu_3)
        self.module_4 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[2]/ret.5(SSD::nndct_conv2d_4)
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[3]/3318(SSD::nndct_relu_5)
        self.module_6 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #SSD::SSD/MaxPool2d[vgg]/ModuleList[4]/3333(SSD::nndct_maxpool_6)
        self.module_7 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[5]/ret.7(SSD::nndct_conv2d_7)
        self.module_8 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[6]/3355(SSD::nndct_relu_8)
        self.module_9 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[7]/ret.9(SSD::nndct_conv2d_9)
        self.module_10 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[8]/3376(SSD::nndct_relu_10)
        self.module_11 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #SSD::SSD/MaxPool2d[vgg]/ModuleList[9]/3391(SSD::nndct_maxpool_11)
        self.module_12 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[10]/ret.11(SSD::nndct_conv2d_12)
        self.module_13 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[11]/3413(SSD::nndct_relu_13)
        self.module_14 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[12]/ret.13(SSD::nndct_conv2d_14)
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[13]/3434(SSD::nndct_relu_15)
        self.module_16 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[14]/ret.15(SSD::nndct_conv2d_16)
        self.module_17 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[15]/3455(SSD::nndct_relu_17)
        self.module_18 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=True) #SSD::SSD/MaxPool2d[vgg]/ModuleList[16]/3470(SSD::nndct_maxpool_18)
        self.module_19 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[17]/ret.17(SSD::nndct_conv2d_19)
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[18]/3492(SSD::nndct_relu_20)
        self.module_21 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[19]/ret.19(SSD::nndct_conv2d_21)
        self.module_22 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[20]/3513(SSD::nndct_relu_22)
        self.module_23 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[21]/ret.21(SSD::nndct_conv2d_23)
        self.module_24 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[22]/3534(SSD::nndct_relu_24)
        self.module_25 = py_nndct.nn.Module('aten::pow') #SSD::SSD/L2Norm[L2Norm]/ret.23(SSD::aten_pow_25)
        self.module_26 = py_nndct.nn.Module('nndct_sum') #SSD::SSD/L2Norm[L2Norm]/ret.25(SSD::nndct_sum_26)
        self.module_27 = py_nndct.nn.sqrt() #SSD::SSD/L2Norm[L2Norm]/ret.27(SSD::nndct_sqrt_27)
        self.module_28 = py_nndct.nn.Add() #SSD::SSD/L2Norm[L2Norm]/ret.29(SSD::nndct_elemwise_add_28)
        self.module_29 = py_nndct.nn.Module('nndct_elemwise_div') #SSD::SSD/L2Norm[L2Norm]/ret.31(SSD::nndct_elemwise_div_29)
        self.module_30 = py_nndct.nn.Module('nndct_unsqueeze') #SSD::SSD/L2Norm[L2Norm]/3553(SSD::nndct_unsqueeze_30)
        self.module_31 = py_nndct.nn.Module('nndct_unsqueeze') #SSD::SSD/L2Norm[L2Norm]/3555(SSD::nndct_unsqueeze_31)
        self.module_32 = py_nndct.nn.Module('nndct_unsqueeze') #SSD::SSD/L2Norm[L2Norm]/3557(SSD::nndct_unsqueeze_32)
        self.module_33 = py_nndct.nn.expand_as() #SSD::SSD/L2Norm[L2Norm]/ret.33(SSD::nndct_expand_as_33)
        self.module_34 = py_nndct.nn.Module('nndct_elemwise_mul') #SSD::SSD/L2Norm[L2Norm]/ret.35(SSD::nndct_elemwise_mul_34)
        self.module_35 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #SSD::SSD/MaxPool2d[vgg]/ModuleList[23]/3576(SSD::nndct_maxpool_35)
        self.module_36 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[24]/ret.37(SSD::nndct_conv2d_36)
        self.module_37 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[25]/3598(SSD::nndct_relu_37)
        self.module_38 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[26]/ret.39(SSD::nndct_conv2d_38)
        self.module_39 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[27]/3619(SSD::nndct_relu_39)
        self.module_40 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[28]/ret.41(SSD::nndct_conv2d_40)
        self.module_41 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[29]/3640(SSD::nndct_relu_41)
        self.module_42 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], ceil_mode=False) #SSD::SSD/MaxPool2d[vgg]/ModuleList[30]/3655(SSD::nndct_maxpool_42)
        self.module_43 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[6, 6], dilation=[6, 6], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[31]/ret.43(SSD::nndct_conv2d_43)
        self.module_44 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[32]/3677(SSD::nndct_relu_44)
        self.module_45 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[vgg]/ModuleList[33]/ret.45(SSD::nndct_conv2d_45)
        self.module_46 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/ReLU[vgg]/ModuleList[34]/3698(SSD::nndct_relu_46)
        self.module_47 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[extras]/ModuleList[0]/ret.47(SSD::nndct_conv2d_47)
        self.module_48 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/3719(SSD::nndct_relu_48)
        self.module_49 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[extras]/ModuleList[1]/ret.49(SSD::nndct_conv2d_49)
        self.module_50 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/3740(SSD::nndct_relu_50)
        self.module_51 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[extras]/ModuleList[2]/ret.51(SSD::nndct_conv2d_51)
        self.module_52 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/3761(SSD::nndct_relu_52)
        self.module_53 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[extras]/ModuleList[3]/ret.53(SSD::nndct_conv2d_53)
        self.module_54 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/3782(SSD::nndct_relu_54)
        self.module_55 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[extras]/ModuleList[4]/ret.55(SSD::nndct_conv2d_55)
        self.module_56 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/3803(SSD::nndct_relu_56)
        self.module_57 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[extras]/ModuleList[5]/ret.57(SSD::nndct_conv2d_57)
        self.module_58 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/3824(SSD::nndct_relu_58)
        self.module_59 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[extras]/ModuleList[6]/ret.59(SSD::nndct_conv2d_59)
        self.module_60 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/3845(SSD::nndct_relu_60)
        self.module_61 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[extras]/ModuleList[7]/ret.61(SSD::nndct_conv2d_61)
        self.module_62 = py_nndct.nn.ReLU(inplace=True) #SSD::SSD/3866(SSD::nndct_relu_62)
        self.module_63 = py_nndct.nn.Conv2d(in_channels=512, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[loc]/ModuleList[0]/ret.63(SSD::nndct_conv2d_63)
        self.module_64 = py_nndct.nn.Module('nndct_permute') #SSD::SSD/ret.65(SSD::nndct_permute_64)
        self.module_65 = py_nndct.nn.Module('nndct_contiguous') #SSD::SSD/ret.67(SSD::nndct_contiguous_65)
        self.module_66 = py_nndct.nn.Conv2d(in_channels=512, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[conf]/ModuleList[0]/ret.69(SSD::nndct_conv2d_66)
        self.module_67 = py_nndct.nn.Module('nndct_permute') #SSD::SSD/ret.71(SSD::nndct_permute_67)
        self.module_68 = py_nndct.nn.Module('nndct_contiguous') #SSD::SSD/ret.73(SSD::nndct_contiguous_68)
        self.module_69 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[loc]/ModuleList[1]/ret.75(SSD::nndct_conv2d_69)
        self.module_70 = py_nndct.nn.Module('nndct_permute') #SSD::SSD/ret.77(SSD::nndct_permute_70)
        self.module_71 = py_nndct.nn.Module('nndct_contiguous') #SSD::SSD/ret.79(SSD::nndct_contiguous_71)
        self.module_72 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[conf]/ModuleList[1]/ret.81(SSD::nndct_conv2d_72)
        self.module_73 = py_nndct.nn.Module('nndct_permute') #SSD::SSD/ret.83(SSD::nndct_permute_73)
        self.module_74 = py_nndct.nn.Module('nndct_contiguous') #SSD::SSD/ret.85(SSD::nndct_contiguous_74)
        self.module_75 = py_nndct.nn.Conv2d(in_channels=512, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[loc]/ModuleList[2]/ret.87(SSD::nndct_conv2d_75)
        self.module_76 = py_nndct.nn.Module('nndct_permute') #SSD::SSD/ret.89(SSD::nndct_permute_76)
        self.module_77 = py_nndct.nn.Module('nndct_contiguous') #SSD::SSD/ret.91(SSD::nndct_contiguous_77)
        self.module_78 = py_nndct.nn.Conv2d(in_channels=512, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[conf]/ModuleList[2]/ret.93(SSD::nndct_conv2d_78)
        self.module_79 = py_nndct.nn.Module('nndct_permute') #SSD::SSD/ret.95(SSD::nndct_permute_79)
        self.module_80 = py_nndct.nn.Module('nndct_contiguous') #SSD::SSD/ret.97(SSD::nndct_contiguous_80)
        self.module_81 = py_nndct.nn.Conv2d(in_channels=256, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[loc]/ModuleList[3]/ret.99(SSD::nndct_conv2d_81)
        self.module_82 = py_nndct.nn.Module('nndct_permute') #SSD::SSD/ret.101(SSD::nndct_permute_82)
        self.module_83 = py_nndct.nn.Module('nndct_contiguous') #SSD::SSD/ret.103(SSD::nndct_contiguous_83)
        self.module_84 = py_nndct.nn.Conv2d(in_channels=256, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[conf]/ModuleList[3]/ret.105(SSD::nndct_conv2d_84)
        self.module_85 = py_nndct.nn.Module('nndct_permute') #SSD::SSD/ret.107(SSD::nndct_permute_85)
        self.module_86 = py_nndct.nn.Module('nndct_contiguous') #SSD::SSD/ret.109(SSD::nndct_contiguous_86)
        self.module_87 = py_nndct.nn.Conv2d(in_channels=256, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[loc]/ModuleList[4]/ret.111(SSD::nndct_conv2d_87)
        self.module_88 = py_nndct.nn.Module('nndct_permute') #SSD::SSD/ret.113(SSD::nndct_permute_88)
        self.module_89 = py_nndct.nn.Module('nndct_contiguous') #SSD::SSD/ret.115(SSD::nndct_contiguous_89)
        self.module_90 = py_nndct.nn.Conv2d(in_channels=256, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[conf]/ModuleList[4]/ret.117(SSD::nndct_conv2d_90)
        self.module_91 = py_nndct.nn.Module('nndct_permute') #SSD::SSD/ret.119(SSD::nndct_permute_91)
        self.module_92 = py_nndct.nn.Module('nndct_contiguous') #SSD::SSD/ret.121(SSD::nndct_contiguous_92)
        self.module_93 = py_nndct.nn.Conv2d(in_channels=256, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[loc]/ModuleList[5]/ret.123(SSD::nndct_conv2d_93)
        self.module_94 = py_nndct.nn.Module('nndct_permute') #SSD::SSD/ret.125(SSD::nndct_permute_94)
        self.module_95 = py_nndct.nn.Module('nndct_contiguous') #SSD::SSD/4195(SSD::nndct_contiguous_95)
        self.module_96 = py_nndct.nn.Conv2d(in_channels=256, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SSD::SSD/Conv2d[conf]/ModuleList[5]/ret.127(SSD::nndct_conv2d_96)
        self.module_97 = py_nndct.nn.Module('nndct_permute') #SSD::SSD/ret.129(SSD::nndct_permute_97)
        self.module_98 = py_nndct.nn.Module('nndct_contiguous') #SSD::SSD/4224(SSD::nndct_contiguous_98)
        self.module_99 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4226(SSD::nndct_shape_99)
        self.module_100 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret.133(SSD::nndct_reshape_100)
        self.module_101 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4235(SSD::nndct_shape_101)
        self.module_102 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret.137(SSD::nndct_reshape_102)
        self.module_103 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4244(SSD::nndct_shape_103)
        self.module_104 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret.141(SSD::nndct_reshape_104)
        self.module_105 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4253(SSD::nndct_shape_105)
        self.module_106 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret.145(SSD::nndct_reshape_106)
        self.module_107 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4262(SSD::nndct_shape_107)
        self.module_108 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret.149(SSD::nndct_reshape_108)
        self.module_109 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4271(SSD::nndct_shape_109)
        self.module_110 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret.153(SSD::nndct_reshape_110)
        self.module_111 = py_nndct.nn.Cat() #SSD::SSD/ret.155(SSD::nndct_concat_111)
        self.module_112 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4284(SSD::nndct_shape_112)
        self.module_113 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret.159(SSD::nndct_reshape_113)
        self.module_114 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4293(SSD::nndct_shape_114)
        self.module_115 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret.163(SSD::nndct_reshape_115)
        self.module_116 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4302(SSD::nndct_shape_116)
        self.module_117 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret.167(SSD::nndct_reshape_117)
        self.module_118 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4311(SSD::nndct_shape_118)
        self.module_119 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret.171(SSD::nndct_reshape_119)
        self.module_120 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4320(SSD::nndct_shape_120)
        self.module_121 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret.175(SSD::nndct_reshape_121)
        self.module_122 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4329(SSD::nndct_shape_122)
        self.module_123 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret.179(SSD::nndct_reshape_123)
        self.module_124 = py_nndct.nn.Cat() #SSD::SSD/ret.181(SSD::nndct_concat_124)
        self.module_125 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4342(SSD::nndct_shape_125)
        self.module_126 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret.185(SSD::nndct_reshape_126)
        self.module_127 = py_nndct.nn.Module('nndct_shape') #SSD::SSD/4352(SSD::nndct_shape_127)
        self.module_128 = py_nndct.nn.Module('nndct_reshape') #SSD::SSD/ret(SSD::nndct_reshape_128)
        self.L2Norm_weight = torch.nn.parameter.Parameter(torch.Tensor(512,))

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(data=1.000000013351432e-10, dtype=torch.float, device='cpu')
        output_module_1 = self.module_1(input=args[0])
        output_module_1 = self.module_2(output_module_1)
        output_module_1 = self.module_3(output_module_1)
        output_module_1 = self.module_4(output_module_1)
        output_module_1 = self.module_5(output_module_1)
        output_module_1 = self.module_6(output_module_1)
        output_module_1 = self.module_7(output_module_1)
        output_module_1 = self.module_8(output_module_1)
        output_module_1 = self.module_9(output_module_1)
        output_module_1 = self.module_10(output_module_1)
        output_module_1 = self.module_11(output_module_1)
        output_module_1 = self.module_12(output_module_1)
        output_module_1 = self.module_13(output_module_1)
        output_module_1 = self.module_14(output_module_1)
        output_module_1 = self.module_15(output_module_1)
        output_module_1 = self.module_16(output_module_1)
        output_module_1 = self.module_17(output_module_1)
        output_module_1 = self.module_18(output_module_1)
        output_module_1 = self.module_19(output_module_1)
        output_module_1 = self.module_20(output_module_1)
        output_module_1 = self.module_21(output_module_1)
        output_module_1 = self.module_22(output_module_1)
        output_module_1 = self.module_23(output_module_1)
        output_module_1 = self.module_24(output_module_1)
        output_module_25 = self.module_25({'self': output_module_1,'exponent': 2})
        output_module_25 = self.module_26(input=output_module_25, dim=1, keepdim=True)
        output_module_25 = self.module_27(input=output_module_25)
        output_module_25 = self.module_28(input=output_module_25, other=output_module_0, alpha=1)
        output_module_29 = self.module_29(input=output_module_1, other=output_module_25)
        output_module_30 = self.module_30(input=self.L2Norm_weight, dim=0)
        output_module_31 = self.module_31(input=output_module_30, dim=2)
        output_module_32 = self.module_32(input=output_module_31, dim=3)
        output_module_32 = self.module_33(input=output_module_32, other=output_module_29)
        output_module_32 = self.module_34(input=output_module_32, other=output_module_29)
        output_module_35 = self.module_35(output_module_1)
        output_module_35 = self.module_36(output_module_35)
        output_module_35 = self.module_37(output_module_35)
        output_module_35 = self.module_38(output_module_35)
        output_module_35 = self.module_39(output_module_35)
        output_module_35 = self.module_40(output_module_35)
        output_module_35 = self.module_41(output_module_35)
        output_module_35 = self.module_42(output_module_35)
        output_module_35 = self.module_43(output_module_35)
        output_module_35 = self.module_44(output_module_35)
        output_module_35 = self.module_45(output_module_35)
        output_module_35 = self.module_46(output_module_35)
        output_module_47 = self.module_47(output_module_35)
        output_module_47 = self.module_48(output_module_47)
        output_module_47 = self.module_49(output_module_47)
        output_module_47 = self.module_50(output_module_47)
        output_module_51 = self.module_51(output_module_47)
        output_module_51 = self.module_52(output_module_51)
        output_module_51 = self.module_53(output_module_51)
        output_module_51 = self.module_54(output_module_51)
        output_module_55 = self.module_55(output_module_51)
        output_module_55 = self.module_56(output_module_55)
        output_module_55 = self.module_57(output_module_55)
        output_module_55 = self.module_58(output_module_55)
        output_module_59 = self.module_59(output_module_55)
        output_module_59 = self.module_60(output_module_59)
        output_module_59 = self.module_61(output_module_59)
        output_module_59 = self.module_62(output_module_59)
        output_module_63 = self.module_63(output_module_32)
        output_module_63 = self.module_64(dims=[0,2,3,1], input=output_module_63)
        output_module_63 = self.module_65(output_module_63)
        output_module_66 = self.module_66(output_module_32)
        output_module_66 = self.module_67(dims=[0,2,3,1], input=output_module_66)
        output_module_66 = self.module_68(output_module_66)
        output_module_69 = self.module_69(output_module_35)
        output_module_69 = self.module_70(dims=[0,2,3,1], input=output_module_69)
        output_module_69 = self.module_71(output_module_69)
        output_module_72 = self.module_72(output_module_35)
        output_module_72 = self.module_73(dims=[0,2,3,1], input=output_module_72)
        output_module_72 = self.module_74(output_module_72)
        output_module_75 = self.module_75(output_module_47)
        output_module_75 = self.module_76(dims=[0,2,3,1], input=output_module_75)
        output_module_75 = self.module_77(output_module_75)
        output_module_78 = self.module_78(output_module_47)
        output_module_78 = self.module_79(dims=[0,2,3,1], input=output_module_78)
        output_module_78 = self.module_80(output_module_78)
        output_module_81 = self.module_81(output_module_51)
        output_module_81 = self.module_82(dims=[0,2,3,1], input=output_module_81)
        output_module_81 = self.module_83(output_module_81)
        output_module_84 = self.module_84(output_module_51)
        output_module_84 = self.module_85(dims=[0,2,3,1], input=output_module_84)
        output_module_84 = self.module_86(output_module_84)
        output_module_87 = self.module_87(output_module_55)
        output_module_87 = self.module_88(dims=[0,2,3,1], input=output_module_87)
        output_module_87 = self.module_89(output_module_87)
        output_module_90 = self.module_90(output_module_55)
        output_module_90 = self.module_91(dims=[0,2,3,1], input=output_module_90)
        output_module_90 = self.module_92(output_module_90)
        output_module_93 = self.module_93(output_module_59)
        output_module_93 = self.module_94(dims=[0,2,3,1], input=output_module_93)
        output_module_93 = self.module_95(output_module_93)
        output_module_96 = self.module_96(output_module_59)
        output_module_96 = self.module_97(dims=[0,2,3,1], input=output_module_96)
        output_module_96 = self.module_98(output_module_96)
        output_module_99 = self.module_99(input=output_module_63, dim=0)
        output_module_100 = self.module_100(input=output_module_63, shape=[output_module_99,-1])
        output_module_101 = self.module_101(input=output_module_69, dim=0)
        output_module_102 = self.module_102(input=output_module_69, shape=[output_module_101,-1])
        output_module_103 = self.module_103(input=output_module_75, dim=0)
        output_module_104 = self.module_104(input=output_module_75, shape=[output_module_103,-1])
        output_module_105 = self.module_105(input=output_module_81, dim=0)
        output_module_106 = self.module_106(input=output_module_81, shape=[output_module_105,-1])
        output_module_107 = self.module_107(input=output_module_87, dim=0)
        output_module_108 = self.module_108(input=output_module_87, shape=[output_module_107,-1])
        output_module_109 = self.module_109(input=output_module_93, dim=0)
        output_module_110 = self.module_110(input=output_module_93, shape=[output_module_109,-1])
        output_module_100 = self.module_111(dim=1, tensors=[output_module_100,output_module_102,output_module_104,output_module_106,output_module_108,output_module_110])
        output_module_112 = self.module_112(input=output_module_66, dim=0)
        output_module_113 = self.module_113(input=output_module_66, shape=[output_module_112,-1])
        output_module_114 = self.module_114(input=output_module_72, dim=0)
        output_module_115 = self.module_115(input=output_module_72, shape=[output_module_114,-1])
        output_module_116 = self.module_116(input=output_module_78, dim=0)
        output_module_117 = self.module_117(input=output_module_78, shape=[output_module_116,-1])
        output_module_118 = self.module_118(input=output_module_84, dim=0)
        output_module_119 = self.module_119(input=output_module_84, shape=[output_module_118,-1])
        output_module_120 = self.module_120(input=output_module_90, dim=0)
        output_module_121 = self.module_121(input=output_module_90, shape=[output_module_120,-1])
        output_module_122 = self.module_122(input=output_module_96, dim=0)
        output_module_123 = self.module_123(input=output_module_96, shape=[output_module_122,-1])
        output_module_113 = self.module_124(dim=1, tensors=[output_module_113,output_module_115,output_module_117,output_module_119,output_module_121,output_module_123])
        output_module_125 = self.module_125(input=output_module_100, dim=0)
        output_module_126 = self.module_126(input=output_module_100, shape=[output_module_125,-1,4])
        output_module_127 = self.module_127(input=output_module_113, dim=0)
        output_module_128 = self.module_128(input=output_module_113, shape=[output_module_127,-1,4])
        return (output_module_126,output_module_128)
