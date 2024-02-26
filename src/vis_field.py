import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.feature as cf
from petrel_client.client import Client
import io, os
import numpy as np
from PIL import Image
class VisField():
    def __init__(self, lon: list, lat: list):
        super().__init__()
        self.lon = lon
        self.lat = lat
    
        self.projection = ccrs.Mercator()
        self.crs = ccrs.PlateCarree()
    def forward_single_image_w_stn(self, data, loss_list, w2k_x, w2k_y, var_name, result_file_name):
        w2k_x = w2k_x.cpu().detach().numpy()
        w2k_y = w2k_y.cpu().detach().numpy()
        data = data.cpu().detach().numpy()
        data_min = np.min(data)
        data_max = np.max(data)
        if var_name == 't2m':
            _cmap = 'RdYlBu_r'
            bar_name = '[K]'
            vmax = 0.05
            vmin = 0.
        elif var_name == 'tp1h':
            _cmap = 'Blues'
            bar_name = '[mm]'
            vmax = 0.08
            vmin = 0.
            data_min = data_min-1.e-3
        elif var_name == 'sp':
            _cmap = 'jet'
            bar_name = '[Pa]'
            vmax = 0.25
            vmin = 0.
        elif var_name == 'u10' or var_name == 'v10':
            bar_name = '[m/s]'
            _cmap = 'seismic'
            vmax = 0.12
            vmin = 0.
        
        p_lon = (w2k_x/27000)*0.25 + 80.
        p_lat = (w2k_y/27000)*0.25 + 18.
        
        plt.figure(dpi=150)
        ax = plt.axes(projection = self.projection, frameon=True)
        ax.set_extent([min(self.lon), max(self.lon), min(self.lat), max(self.lat)], crs=self.crs)
        
        if data_min<0 and data_max>0:
            data_min = -max(np.abs(data_min), np.abs(data_max))
            data_max = max(np.abs(data_min), np.abs(data_max))
        # ax.set_title(f"Input Field of {var_name}")
        im1 = ax.imshow(data,cmap=_cmap,extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], alpha=0.5,vmin=data_min, vmax=data_max, transform=self.crs)
        #img_bar = plt.colorbar(im1, orientation='horizontal', shrink=0.66)
        #img_bar.set_label(bar_name)
        #loss_min, loss_max = np.percentile(loss_list, [5,95])
        norm = mcolors.Normalize(vmin=vmin,vmax=vmax)
        scatter = plt.scatter(p_lon[0].tolist(), p_lat[0].tolist(), marker='o', s=20, c=loss_list, cmap='hot', edgecolors='k', transform=self.crs, norm=norm)
        #sc_bar = plt.colorbar(scatter, shrink=0.66, orientation='horizontal')
        #sc_bar.set_label("Norm-MSE")
        plt.savefig(result_file_name,bbox_inches='tight')
        plt.close()


    def forward_single_image(self, data, result_file_name):
        # if var_name == 't2m':
        #     _cmap = 'RdYlBu'
        # elif var_name == 'tp1h':
        #     _cmap = 'Blues'
        # elif var_name == 'sp':
        #     _cmap = 'jet'
        # elif var_name == 'u10' or var_name == 'v10':
        #     _cmap = 'seismic'
        _cmap = 'RdYlBu_r'

        plt.figure(dpi=150)
        ax = plt.axes(projection = self.projection, frameon=True)
        # gl = ax.gridlines(crs=self.crs, draw_labels=True,
        #                         linewidth=.6, color='gray',
        #                         alpha=0.5, linestyle='-.')
        # gl.xlabel_style = {"size": 7}
        # gl.ylabel_style = {"size": 7}
        ax.set_extent([min(self.lon), max(self.lon), min(self.lat), max(self.lat)], crs=self.crs)
        data_min = np.min(data)
        data_max = np.max(data)
        if data_min<0 and data_max>0:
            data_min = -max(np.abs(data_min), np.abs(data_max))
            data_max = max(np.abs(data_min), np.abs(data_max))
        # ax.set_title(f"Input Field of {var_name}")
        im1 = ax.imshow(data,cmap=_cmap,extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        # plt.colorbar(im1, ax=ax, shrink=0.5)
        plt.savefig(result_file_name,bbox_inches='tight')
        plt.close()

    def forward(self, data_input, data_pred, data_baseline, data_gt, var_name, result_file_name):
        # generate basemap
        # plt.figure(dpi=150)

        if var_name == 't2m':
            _cmap = 'RdYlBu_r'
        elif var_name == 'tp1h':
            _cmap = 'Blues'
        elif var_name == 'sp':
            _cmap = 'jet'
        elif var_name == 'u10' or var_name == 'v10':
            _cmap = 'seismic'
        fig, axes = plt.subplots(2, 3, subplot_kw={'projection':self.projection},figsize=(30,10))
        for i in range(2):
            for j in range(3):
                gl = axes[i,j].gridlines(crs=self.crs, draw_labels=True,
                                linewidth=.6, color='gray',
                                alpha=0.5, linestyle='-.')
                gl.xlabel_style = {"size": 7}
                gl.ylabel_style = {"size": 7}
                axes[i,j].set_extent([min(self.lon), max(self.lon), min(self.lat), max(self.lat)], crs=self.crs)
        
        # ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
        # ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.3)

        data_min = min(np.min(data_pred), np.min(data_gt))
        data_max = max(np.max(data_pred), np.max(data_gt))
        if data_min<0 and data_max>0:
            data_min = -max(np.abs(data_min), np.abs(data_max))
            data_max = max(np.abs(data_min), np.abs(data_max))
        
        
        axes[0,0].set_title(f"Input Field of {var_name}")
        im1 = axes[0,0].imshow(data_input,cmap=_cmap,extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        plt.colorbar(im1, ax=axes[0,0], shrink=0.5)
        axes[1,0].set_title(f"Target Field of {var_name}")
        im2 = axes[1,0].imshow(data_gt,cmap=_cmap,extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        plt.colorbar(im2, ax=axes[1,0], shrink=0.5)
        axes[0,1].set_title(f"Pred Field of {var_name}")
        im3 = axes[0,1].imshow(data_pred,cmap=_cmap,extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        plt.colorbar(im3, ax=axes[0,1], shrink=0.5)
        axes[1,1].set_title(f"Baseline Field of {var_name}")
        im4 = axes[1,1].imshow(data_baseline,cmap=_cmap,extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        plt.colorbar(im4, ax=axes[1,1], shrink=0.5)
        
        ape1 = np.abs((data_gt-data_pred)/(np.abs(data_gt)+1e-4))
        axes[0,2].set_title(f"Pred Error Field of {var_name}")
        im5 = axes[0,2].imshow(ape1, cmap='OrRd',extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=np.percentile(ape1,0.01), vmax=np.percentile(ape1,99.99), transform=self.crs)
        plt.colorbar(im5, ax=axes[0,2], shrink=0.5)

        ape2 = np.abs((data_gt-data_baseline)/(np.abs(data_gt)+1e-4))
        axes[1,2].set_title(f"Baseline Error Field of {var_name}")
        im5 = axes[1,2].imshow(ape2, cmap='OrRd',extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=np.percentile(ape1,0.01), vmax=np.percentile(ape1,99.99), transform=self.crs)
        plt.colorbar(im5, ax=axes[1,2], shrink=0.5)
        plt.savefig(result_file_name,bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    lon_range = [72., 136.]
    lat_range = [18., 54.]
    client = Client(conf_path="/mnt/petrelfs/liuzili/petreloss.conf")
    data_path = 'cluster1:s3://pretrained_models/TIGGE/NCEP_1d/2021/GFS_2021-01-01-00-00-00_f000_rio.tiff'
    with io.BytesIO(client.get(data_path)) as f:
        # data = np.load(f)[144:288, 320:544]
        data = np.array(Image.open(f))
    # data_path = '/mnt/petrelfs/liuzili/data/PINNs_draw/results/GFS_2021-01-01-00-00-00_f000_v10.tiff'
        
        # data = np.array(Image.open(data_path))
    result_path = '/mnt/petrelfs/liuzili/data/PINNs_draw/vis'
    

    VisUtil = VisField(lon_range, lat_range)
    # data_input = data[::4,::4]
    data = np.flipud(data)
    filename = os.path.join(result_path, 'Pred_GFS_rio.png')
    VisUtil.forward_single_image(data,'v10', filename)