import torch
import torch.nn as nn
import torch.nn.functional as F

"Mobile-Optimized Enhanced EMA Attention Module"

class MobileChannelInteraction(nn.Module):
    """移动端优化的通道间交互模块"""
    def __init__(self, channels, reduction=4):
        super(MobileChannelInteraction, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # 轻量化的通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 简化的通道注意力网络
        self.channel_att = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        # 轻量化的通道间交互
        self.channel_interaction = nn.Conv2d(channels, channels, 1, groups=channels//8, bias=False)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # 全局特征提取
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        # 通道注意力
        channel_att = self.channel_att(avg_out + max_out).view(b, c, 1, 1)
        
        # 通道间交互
        channel_interaction = self.channel_interaction(x)
        
        # 残差连接
        out = x + x * channel_att + channel_interaction * (1 - channel_att)
        
        return out

class MobileCrossSpatialLearning(nn.Module):
    """移动端优化的跨空间学习模块"""
    def __init__(self, channels, groups=32):
        super(MobileCrossSpatialLearning, self).__init__()
        self.channels = channels
        self.groups = groups
        self.group_channels = channels // groups
        
        # 轻量化的多尺度特征提取
        self.spatial_conv1 = nn.Conv2d(self.group_channels, self.group_channels, 1, bias=False)
        self.spatial_conv3 = nn.Conv2d(self.group_channels, self.group_channels, 3, padding=1, groups=self.group_channels//4, bias=False)
        
        # 简化的空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.group_channels * 2, self.group_channels, 1, bias=False),
            nn.BatchNorm2d(self.group_channels),
            nn.Sigmoid()
        )
        
        # 轻量化的跨空间交互
        self.cross_spatial = nn.Conv2d(self.group_channels, self.group_channels, 1, groups=self.group_channels//2, bias=False)
        
    def forward(self, x):
        # 多尺度空间特征
        feat1 = self.spatial_conv1(x)
        feat3 = self.spatial_conv3(x)
        
        # 多尺度特征融合
        multi_scale_feat = torch.cat([feat1, feat3], dim=1)
        spatial_att = self.spatial_attention(multi_scale_feat)
        
        # 跨空间交互
        cross_spatial = self.cross_spatial(x)
        
        # 残差连接
        out = x + x * spatial_att + cross_spatial * (1 - spatial_att)
        
        return out

class MobileEMA(nn.Module):
    """移动端优化的EMA注意力模块"""
    def __init__(self, channels, c2=None, factor=32, reduction=4):
        super(MobileEMA, self).__init__()
        self.groups = factor
        self.channels = channels
        assert channels // self.groups > 0
        
        # 坐标注意力模块（保持原有功能但优化）
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        
        # 轻量化的卷积层
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, 1, bias=False)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, 3, padding=1, 
                                 groups=channels // self.groups // 2, bias=False)
        
        # 移动端优化的通道间交互
        self.channel_interaction = MobileChannelInteraction(channels, reduction)
        
        # 移动端优化的跨空间学习
        self.cross_spatial_learning = MobileCrossSpatialLearning(channels, self.groups)
        
        # 轻量化的特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        #(B,C,H,W)
        b, c, h, w = x.size()
        
        # 1. 通道间交互增强
        channel_enhanced_x = self.channel_interaction(x)
        
        # 2. 坐标注意力模块（优化版）
        group_x = channel_enhanced_x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        
        # 1×1分支和3×3分支
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        
        # 3. 跨空间学习（移动端优化版）
        x1_reshaped = x1.reshape(b, c, h, w)
        x2_reshaped = x2.reshape(b, c, h, w)
        
        # 对每个组分别处理
        x1_enhanced_list = []
        x2_enhanced_list = []
        
        for i in range(self.groups):
            start_ch = i * (c // self.groups)
            end_ch = (i + 1) * (c // self.groups)
            
            x1_group = x1_reshaped[:, start_ch:end_ch, :, :]
            x2_group = x2_reshaped[:, start_ch:end_ch, :, :]
            
            x1_enhanced_group = self.cross_spatial_learning(x1_group)
            x2_enhanced_group = self.cross_spatial_learning(x2_group)
            
            x1_enhanced_list.append(x1_enhanced_group)
            x2_enhanced_list.append(x2_enhanced_group)
        
        x1_enhanced = torch.cat(x1_enhanced_list, dim=1).reshape(b * self.groups, -1, h, w)
        x2_enhanced = torch.cat(x2_enhanced_list, dim=1).reshape(b * self.groups, -1, h, w)
        
        # 4. 增强的跨空间学习
        x11 = self.softmax(self.agp(x1_enhanced).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2_enhanced.reshape(b * self.groups, c // self.groups, -1)
        y1 = torch.matmul(x11, x12)
        
        x21 = self.softmax(self.agp(x2_enhanced).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1_enhanced.reshape(b * self.groups, c // self.groups, -1)
        y2 = torch.matmul(x21, x22)
        
        # 5. 空间权重生成
        weights = (y1 + y2).reshape(b * self.groups, 1, h, w)
        weights_ = weights.sigmoid()
        spatial_enhanced = (group_x * weights_).reshape(b, c, h, w)
        
        # 6. 特征融合
        fused_feat = self.feature_fusion(spatial_enhanced)
        
        # 7. 最终融合
        final_weights = self.final_fusion(fused_feat)
        
        # 8. 输出
        out = channel_enhanced_x * final_weights + spatial_enhanced * (1 - final_weights)
        
        return out

class UltraLightEMA(nn.Module):
    """超轻量化EMA模块，适合资源受限环境"""
    def __init__(self, channels, factor=32):
        super(UltraLightEMA, self).__init__()
        self.channels = channels
        self.groups = factor
        
        # 极简的通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        # 极简的空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 极简的跨空间学习
        self.cross_spatial = nn.Conv2d(channels, channels, 1, groups=channels//self.groups)
        
    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_att(x)
        x_channel = x * channel_att
        
        # 空间注意力
        spatial_att = self.spatial_att(x_channel)
        x_spatial = x_channel * spatial_att
        
        # 跨空间学习
        cross_spatial = self.cross_spatial(x_spatial)
        
        # 残差连接
        out = x + cross_spatial
        
        return out

if __name__ == '__main__':
    # 测试移动端优化的EMA模块
    print("测试移动端优化的EMA模块...")
    
    # 测试输入
    input_tensor = torch.randn(2, 512, 7, 7)
    print(f"输入形状: {input_tensor.shape}")
    
    # 移动端优化EMA
    mobile_ema = MobileEMA(channels=512)
    output_mobile = mobile_ema(input_tensor)
    print(f"移动端EMA输出形状: {output_mobile.shape}")
    
    # 超轻量化EMA
    ultra_light_ema = UltraLightEMA(channels=512)
    output_ultra = ultra_light_ema(input_tensor)
    print(f"超轻量化EMA输出形状: {output_ultra.shape}")
    
    # 计算参数量
    mobile_params = sum(p.numel() for p in mobile_ema.parameters())
    ultra_params = sum(p.numel() for p in ultra_light_ema.parameters())
    
    print(f"移动端EMA参数量: {mobile_params:,}")
    print(f"超轻量化EMA参数量: {ultra_params:,}")
    print(f"参数量差异: {mobile_params - ultra_params:,}")
    
    # 计算FLOPs（近似）
    def calculate_flops(model, input_shape):
        model.eval()
        with torch.no_grad():
            x = torch.randn(input_shape)
            # 这里只是示例，实际FLOPs计算需要更复杂的工具
            return sum(p.numel() for p in model.parameters())
    
    print(f"移动端EMA FLOPs (近似): {calculate_flops(mobile_ema, (1, 512, 7, 7)):,}")
    print(f"超轻量化EMA FLOPs (近似): {calculate_flops(ultra_light_ema, (1, 512, 7, 7)):,}")
    
    print("测试完成!")

