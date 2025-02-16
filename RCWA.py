import torch
import numpy as np  
import torcwa
import matplotlib.pyplot as plt
import Materials

class RCWA:
    def __init__(
        self,
        device= torch.device('cpu'),
        shape_type='rectangle',
        harmonic_order=7,
        wavelength=940.,
        period=400.,
        substrate_material='SiO2.txt',
        metasurface_thickness=500.,
        metasurface_material='Si.txt',
        slab_thickness=135.,
        slab_material='SiO2.txt',
        filling_thickness=0.,
        filling_material='SiO2.txt',
        output_material='air.txt',
        # shape parameters
        Wx=None,
        Wy=None,
        theta=None,
        Rx=None,
        Ry=None,
        R=None,
        hollow_W=None,
        hollow_R=None
    ):
        self.device = device
        self.shape_type = shape_type
        self.harmonic_order = harmonic_order
        self.wavelength = wavelength
        self.period = period
        self.metasurface_thickness = metasurface_thickness
        self.metasurface_material = metasurface_material
        self.substrate_material = substrate_material
        self.slab_thickness = slab_thickness
        self.slab_material = slab_material
        self.filling_material = filling_material
        self.filling_thickness = filling_thickness
        self.output_material = output_material
        # shape parameters
        self.Wx = Wx
        self.Wy = Wy
        if theta is not None:
            self.theta = theta/180*np.pi
        else:
            self.theta = theta
        self.Rx = Rx
        self.Ry = Ry
        self.R  = R
        self.hollow_W = hollow_W
        self.hollow_R = hollow_R

    def show_structure(self):
        """
        在這裡實作或呼叫建構結構所需的程式碼。
        """
        geo_dtype = torch.float32
        device = self.device

        # Simulation environment
        # light
        lamb0 = torch.tensor(self.wavelength,dtype=geo_dtype,device=device)    # nm

        # material
        silicon_eps = Materials.Material.forward(wavelength=lamb0, name=self.metasurface_material)**2
        filling_eps = Materials.Material.forward(wavelength=lamb0, name=self.filling_material)**2

        # geometry
        L = [self.period, self.period]            # nm / nm
        torcwa.rcwa_geo.dtype = geo_dtype
        torcwa.rcwa_geo.device = device
        torcwa.rcwa_geo.Lx = L[0]
        torcwa.rcwa_geo.Ly = L[1]
        torcwa.rcwa_geo.nx = 300
        torcwa.rcwa_geo.ny = 300
        torcwa.rcwa_geo.grid()
        torcwa.rcwa_geo.edge_sharpness = 1000.

        x_axis = torcwa.rcwa_geo.x.cpu()
        y_axis = torcwa.rcwa_geo.y.cpu()
        if self.shape_type == 'rectangle':
            layer0_geometry = torcwa.rcwa_geo.rectangle(Wx=self.Wx,Wy=self.Wy,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta)
        elif self.shape_type == 'ellipse':
            layer0_geometry = torcwa.rcwa_geo.ellipse(Rx=self.Rx/2,Ry=self.Ry/2,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta)
        elif self.shape_type == 'circle':
            layer0_geometry = torcwa.rcwa_geo.circle(R=self.R/2,Cx=L[0]/2.,Cy=L[1]/2.)
        elif self.shape_type == 'rhombus':
            layer0_geometry = torcwa.rcwa_geo.rhombus(Wx=self.Wx,Wy=self.Wy,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta)
        elif self.shape_type == 'square':
            layer0_geometry = torcwa.rcwa_geo.square(W=self.Wx,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta)
        elif self.shape_type == 'cross':
            layer0_geometry_A = torcwa.rcwa_geo.rectangle(Wx=self.Wx,Wy=self.Wy,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta+0)
            layer0_geometry_B = torcwa.rcwa_geo.rectangle(Wx=self.Wx,Wy=self.Wy,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta+np.pi/2)
            layer0_geometry = torcwa.rcwa_geo.union(layer0_geometry_A,layer0_geometry_B)
        #elif self.shape_type == 'cross':
        
        #    layer0_geometry_A = torcwa.rcwa_geo.rectangle(Wx=self.Wx,Wy=self.Wy,Cx=L[0]/2. , Cy=L[1]/2., theta=self.theta+0)
        #    layer0_geometry_B = torcwa.rcwa_geo.rectangle(Wx=self.Wy,Wy=self.Wx,Cx=L[0]/2. -self.Wx/2 , Cy=L[1]/2. , theta=self.theta+0)
        #    layer0_geometry_C = torcwa.rcwa_geo.rectangle(Wx=self.Wy,Wy=self.Wx,Cx=L[0]/2. + self.Wx/2 ,Cy=L[1]/2. , theta=self.theta+0)
        #    layer0_geometry = torcwa.rcwa_geo.union(layer0_geometry_A, layer0_geometry_B)
        #    layer0_geometry = torcwa.rcwa_geo.union(layer0_geometry, layer0_geometry_C)
        elif self.shape_type == 'hollow_square':
            layer0_geometry_A = torcwa.rcwa_geo.square(W=self.Wx,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta)
            layer0_geometry_B = torcwa.rcwa_geo.square(W=self.hollow_W,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta)
            layer0_geometry = torcwa.rcwa_geo.difference(layer0_geometry_A,layer0_geometry_B)
        elif self.shape_type == 'hollow_circle':
            layer0_geometry_A = torcwa.rcwa_geo.circle(R=self.R/2,Cx=L[0]/2.,Cy=L[1]/2.)
            layer0_geometry_B = torcwa.rcwa_geo.circle(R=self.hollow_R/2,Cx=L[0]/2.,Cy=L[1]/2.)
            layer0_geometry = torcwa.rcwa_geo.difference(layer0_geometry_A,layer0_geometry_B)
        layer0_eps = layer0_geometry*silicon_eps + filling_eps*(1.-layer0_geometry)
        figure, ax = plt.subplots()
        plt.imshow(torch.transpose(torch.real(layer0_eps),-2,-1).cpu(),origin='lower',extent=[x_axis[0],x_axis[-1],y_axis[0],y_axis[-1]])
        plt.title('Layer 0')
        plt.xlim([0,L[0]])
        plt.xlabel('x (nm)')
        plt.ylim([0,L[1]])
        plt.ylabel('y (nm)')
        plt.title('permittivity')
        plt.colorbar()
        return figure, ax

    def get_Sparameter(self):
        """
        在此實作 RCWA 計算部分，回傳 Transmission 和 Phase。
        這裡先回傳固定值做示範，可自行替換成真實演算法計算結果。
        """
        # Hardware
        # If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
        # If you need accurate operation, you have to disable the flag below.
        #torch.backends.cuda.matmul.allow_tf32 = False
        sim_dtype = torch.complex64
        geo_dtype = torch.float32
        device = self.device

        # Simulation environment
        # light
        lamb0 = torch.tensor(self.wavelength,dtype=geo_dtype,device=device)    # nm
        inc_ang = 0.*(np.pi/180)                    # radian
        azi_ang = 0.*(np.pi/180)                    # radian

        # material
        slab_eps = Materials.Material.forward(wavelength=lamb0, name=self.slab_material)**2
        substrate_eps = Materials.Material.forward(wavelength=lamb0, name=self.substrate_material)**2
        silicon_eps = Materials.Material.forward(wavelength=lamb0, name=self.metasurface_material)**2
        filling_eps = Materials.Material.forward(wavelength=lamb0, name=self.filling_material)**2
        output_eps = Materials.Material.forward(wavelength=lamb0, name=self.output_material)**2
        # geometry
        L = [self.period, self.period]            # nm / nm
        torcwa.rcwa_geo.dtype = geo_dtype
        torcwa.rcwa_geo.device = device
        torcwa.rcwa_geo.Lx = L[0]
        torcwa.rcwa_geo.Ly = L[1]
        torcwa.rcwa_geo.nx = 300
        torcwa.rcwa_geo.ny = 300
        torcwa.rcwa_geo.grid()
        torcwa.rcwa_geo.edge_sharpness = 1000.

        if self.shape_type == 'rectangle':
            layer0_geometry = torcwa.rcwa_geo.rectangle(Wx=self.Wx,Wy=self.Wy,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta)
        elif self.shape_type == 'ellipse':
            layer0_geometry = torcwa.rcwa_geo.ellipse(Rx=self.Rx/2,Ry=self.Ry/2,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta)
        elif self.shape_type == 'circle':
            layer0_geometry = torcwa.rcwa_geo.circle(R=self.R/2,Cx=L[0]/2.,Cy=L[1]/2.)
        elif self.shape_type == 'rhombus':
            layer0_geometry = torcwa.rcwa_geo.rhombus(Wx=self.Wx, Wy=self.Wy,Cx=L[0]/2., Cy=L[1]/2., theta=self.theta)
        elif self.shape_type == 'square':
            layer0_geometry = torcwa.rcwa_geo.square(W=self.Wx, Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta)
        elif self.shape_type == 'cross':
            layer0_geometry_A = torcwa.rcwa_geo.rectangle(Wx=self.Wx,Wy=self.Wy,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta+0)
            layer0_geometry_B = torcwa.rcwa_geo.rectangle(Wx=self.Wx,Wy=self.Wy,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta+np.pi/2)
            layer0_geometry = torcwa.rcwa_geo.union(layer0_geometry_A,layer0_geometry_B)
        elif self.shape_type == 'hollow_square':
            layer0_geometry_A = torcwa.rcwa_geo.square(W=self.Wx,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta)
            layer0_geometry_B = torcwa.rcwa_geo.square(W=self.hollow_W,Cx=L[0]/2.,Cy=L[1]/2., theta=self.theta)
            layer0_geometry = torcwa.rcwa_geo.difference(layer0_geometry_A,layer0_geometry_B)
        elif self.shape_type == 'hollow_circle':
            layer0_geometry_A = torcwa.rcwa_geo.circle(R=self.R/2,Cx=L[0]/2.,Cy=L[1]/2.)
            layer0_geometry_B = torcwa.rcwa_geo.circle(R=self.hollow_R/2,Cx=L[0]/2.,Cy=L[1]/2.)
            layer0_geometry = torcwa.rcwa_geo.difference(layer0_geometry_A,layer0_geometry_B)
        
        # layers
        layer0_thickness = self.metasurface_thickness
        filling_thickness = self.filling_thickness
        slab_thickness =  self.slab_thickness
        # Generate and perform simulation
        order_N = self.harmonic_order
        order = [order_N,order_N]
        sim = torcwa.rcwa(freq=1/lamb0,order=order,L=L,dtype=sim_dtype,device=device)
        sim.add_input_layer(eps=substrate_eps)
        sim.add_output_layer(eps=output_eps)
        sim.set_incident_angle(inc_ang=inc_ang,azi_ang=azi_ang)
        sim.add_layer(thickness=self.slab_thickness,eps=slab_eps)
        layer0_eps = layer0_geometry*silicon_eps + filling_eps*(1.-layer0_geometry)
        sim.add_layer(thickness=layer0_thickness,eps=layer0_eps)
        sim.add_layer(thickness=filling_thickness,eps=filling_eps)
        sim.solve_global_smatrix()
        txx = sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='xx',ref_order=[0,0])
        txy = sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='xy',ref_order=[0,0])
        tyx = sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='yx',ref_order=[0,0])
        tyy = sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='yy',ref_order=[0,0])
        return txx,txy,tyx,tyy