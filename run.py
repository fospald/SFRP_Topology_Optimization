# TODO:
# - plot objective i over sum of step lengths up to iteration i
# - Alberto Paganini kucken
# - look at https://github.com/SCIInstitute/SCI-Solver_Eikonal
# - support for multiple loadcases
# - Dirichlet bc for rho (fixed regions)
# - http://computation.llnl.gov/projects/glvis-finite-element-visualization
# - http://mfem.org/

DEBUGX = True
DEBUGX = False

#from __future__ import division  # DON'T use this, it produces wrong results!

#from block import *
from dolfin import *
import ufl.algorithms.ad
import mshr
import inspect
import traceback
import sys
import hashlib
import numpy as np
import os, time
import traceback
import scipy, scipy.optimize, scipy.sparse, scipy.misc

try:
	raise NotImplemented() # vtk causes MPI runtime error (on my notebook)
	import vtk
	have_vtk = True
except:
	have_vtk = False
	print "WARNING: VTK not found"

from matplotlib.backends import qt4_compat
use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import rcParams
from matplotlib.backend_bases import cursors
import matplotlib.tri as mtri
import matplotlib.cm
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
	import eikonal2d
	have_eikonal2d = True
except:
	have_eikonal2d = False
	print "WARNING: eikonal2d not found"

rcParams.update({'figure.autolayout': True})


np.set_printoptions(precision=3, linewidth=10000)


# additional UFL functions

def eps(u):
	return sym(grad(u))

def ediff(e, x):
	return ufl.algorithms.ad.expand_derivatives(diff(e, x))

def bisect(func, a, b, rtol=1e-12, xtol=1e-12):

	fa = func(a)
	fb = func(b)

	while True:

		c = 0.5*(a+b)
		fc = func(c)

		#print "bisection lambda=%g, f=%g" % (c, fc)

		if abs(fc) < rtol or abs(a-b) < xtol:
			break

		if fa*fb > 0:
			print a, b, fa, fb
			raise RuntimeError("f(a) and f(b) have the same sign")

		if (fc < 0):
			a = c
			fa = fc
		else:
			b = c
			fb = fc

	return c



class MaterialModel(object):
	def __init__(self):
		self.params = []
	# return strass
	def sigma(self, eps, gdim):
		raise NotImplementedError()


class SimpleMaterialModel(MaterialModel):
	def __init__(self):
		app = QtGui.QApplication.instance()
		self.nu = Constant(app.parameters["nu"])
		self.E = Constant(app.parameters["E"])
		self.params = [self.nu]
	def sigma(self, eps, gdim):
		return (self.E/(1+self.nu))*(eps + (self.nu/(1-2*self.nu))*tr(eps)*Identity(gdim))


class IsotropicMaterialModel(MaterialModel):
	def __init__(self):
		app = QtGui.QApplication.instance()
		s = app.parameters["anisotropy_scale"]
		self.c = [1.721, 1.894, -0.0002487*s, 0.1472*s, 1.355*s]
		self.mu = Constant(0.5*self.c[0] + self.c[3]/3.0 + self.c[4]/15.0)
		self.lmbda = Constant(self.c[1] + 2*self.c[2]/3.0 + self.c[4]/15.0)
		self.params = [self.mu, self.lmbda]
	def sigma(self, eps, gdim):
		return 2*(self.mu*eps) + self.lmbda*tr(eps)*Identity(gdim)


class ConstantTransverseIsotropicMaterialModel(MaterialModel):
	def __init__(self):
		app = QtGui.QApplication.instance()
		s = app.parameters["anisotropy_scale"]
		self.c = [1.721, 1.894, -0.0002487*s, 0.1472*s, 1.355*s]
		self.c = [Constant(c) for c in self.c]
		theta = app.parameters["fo_angle"]*np.pi/180.0
		self.p2 = Constant([np.cos(theta), np.sin(theta)])
		self.p3 = Constant([np.cos(theta), np.sin(theta), 0.0])
		self.params = self.c
	def sigma(self, eps, gdim):
		I = Identity(gdim)
		tr_eps = tr(eps)
		p = self.p2 if gdim == 2 else self.p3
		p2 = outer(p, p)
		p2eps = inner(p2, eps)
		return self.c[0]*eps + self.c[1]*tr_eps*I + self.c[2]*(p2*tr_eps + p2eps*I) + self.c[3]*(dot(p2, eps) + dot(eps, p2)) + self.c[4]*p2*p2eps

class FO_Expr(Expression):

	def __init__(self, vtk_file):

		reader = vtk.vtkGenericDataObjectReader()
		reader.SetFileName(vtk_file)
		reader.ReadAllScalarsOn()
		reader.ReadAllVectorsOn()
		reader.ReadAllTensorsOn()
		reader.Update()

		dataset = reader.GetOutput()
		celldata = dataset.GetCellData()

		self.celllocator = vtk.vtkCellLocator()
		self.celllocator.SetDataSet(dataset)
		self.celllocator.BuildLocator()

		# point used for evaluation
		self.data_EV = celldata.GetArray("eigenVectorsA2")
		self.data_EW = celldata.GetArray("eigenValuesA2")

		self.cell = vtk.vtkGenericCell()
		self.cellId = vtk.mutable(0)
		self.subId = vtk.mutable(0)
		self.closestPointDist2 = vtk.mutable(0.0)
		self.closestPoint = [0.0, 0.0, 0.0]

	def eval(self, value, x):
		#cell = self.mesh.closest_cell(x
		value[0] = 1.0
		value[1] = 0.0

		# find closest point and cell in mesh
		point = [x[0], x[1], 0.05]
		self.celllocator.FindClosestPoint(point, self.closestPoint, self.cell, self.cellId, self.subId, self.closestPointDist2)

		# get (constant) cell data
		V = np.zeros((3,3))
		for j in range(3):
			for i in range(2):
				V[i,j] = self.data_EV.GetComponent(self.cellId, i*3 + j)
			n = np.linalg.norm(V[:,j])
			if n == 0:
				print "WARNING: n == 0"
			V[:,j] /= n

		D = np.zeros((3,3))
		D3 = self.data_EW.GetComponent(self.cellId, 2)
		D[0,0] = self.data_EW.GetComponent(self.cellId, 0) + 0.5*D3
		D[1,1] = self.data_EW.GetComponent(self.cellId, 1) + 0.5*D3
		D /= D[0,0] + D[1,1]

		A = np.dot(np.dot(V, D), V.T)
		
		for j in range(2):
			for i in range(2):
				value[i*2+j] = A[i,j]
		
		#print A
		#sys.exit(0)


	def value_shape(self):
		return (2,2)

class FileTransverseIsotropicMaterialModel(MaterialModel):
	def __init__(self):
		app = QtGui.QApplication.instance()
		self.c = [1.721, 1.894, -0.0002487, 0.1472, 1.355]
		self.c = [Constant(c) for c in self.c]
		self.params = self.c
	def sigma(self, eps, A, ascale, iscale, gdim):
		I = Identity(gdim)
		tr_eps = tr(eps)
		p2 = A*(1-iscale) + I*iscale/gdim
		p2eps = inner(p2, eps)
		return self.c[0]*eps + self.c[1]*tr_eps*I + ascale*(self.c[2]*(p2*tr_eps + p2eps*I) + self.c[3]*(dot(p2, eps) + dot(eps, p2)) + self.c[4]*p2*p2eps)

class TransverseIsotropicMaterialModel(MaterialModel):
	def __init__(self):
		app = QtGui.QApplication.instance()
		self.c = [1.721, 1.894, -0.0002487, 0.1472, 1.355]
		self.c = [Constant(c) for c in self.c]
		self.params = self.c
	def sigma(self, eps, p, ascale, iscale, gdim):
		I = Identity(gdim)
		tr_eps = tr(eps)
		p2 = outer(p, p)*(1-iscale) + I*iscale/gdim
		p2eps = inner(p2, eps)
		return self.c[0]*eps + self.c[1]*tr_eps*I + ascale*(self.c[2]*(p2*tr_eps + p2eps*I) + self.c[3]*(dot(p2, eps) + dot(eps, p2)) + self.c[4]*p2*p2eps)
	def dsigma(self, eps, p, dp, ascale, iscale, gdim):
		I = Identity(gdim)
		tr_eps = tr(eps)
		p2 = outer(p, p)*(1-iscale) + I*iscale/gdim
		dp2 = (outer(dp, p) + outer(p, dp))*(1-iscale)
		p2eps = inner(p2, eps)
		dp2eps = inner(dp2, eps)
		return ascale*(self.c[2]*(dp2*tr_eps + dp2eps*I) + self.c[3]*(dot(dp2, eps) + dot(eps, dp2)) + self.c[4]*(p2*dp2eps + dp2*p2eps))
	def B(self, eps, p, x, ascale, iscale, gdim):
		I = Identity(gdim)
		p2 = outer(p, p)
		dp = dot(I - p2, x)
		return self.dsigma(eps, p, dp, ascale, iscale, gdim)
	def Bstar(self, eps, p, x, ascale, iscale, gdim):
		for i in range(gdim):
			ei = unit_vector(i, gdim)
			s = inner(self.B(eps, p, ei, ascale, iscale, gdim), x)*ei
			r = s if i == 0 else (r + s)
		return r


class Param(object):
	def __init__(self, **kwargs):
		for key, value in kwargs.iteritems():
			setattr(self, key, value)

class FunctionExpression(Expression):
	def eval_cell(self, value, x, ufc_cell):
		self.f.eval(value, x)


class TopologyOptimizationLoadcase(object):

	# return single or list of DirichletBC instances for the displacement
	def getDirichletBC(self, V):
		raise []

	# return Function or Expression on V for the volume gravity g0
	def getVolumeForce(self, V):
		return Constant([0]*self.problem.mesh_gdim)

	# return Function or Expression on V for the volume forces g1
	def getVolumeGravity(self, V):
		return Constant([0]*self.problem.mesh_gdim)

	# return Function or Expression on V for the boundary tractions
	def getBoundaryTraction(self, V):
		return Constant([0]*self.problem.mesh_gdim)

	# return point sources
	def getPointSources(self, V):
		return []

	# return weight of loadcase in objective
	def getWeigth(self):
		return 1.0


class ComplianceTopologyOptimizationLoadcase(TopologyOptimizationLoadcase):

	def __init__(self, problem):

		app = QtGui.QApplication.instance()

		self.problem = problem
		self.weight = Constant(self.getWeigth())

		# init Dirichlet bc
		self.bc_u = self.getDirichletBC(problem.V_u)

		# check if bc are homogeneous
		self.homogeneous = True
		for bc in self.bc_u:
			if not np.all(np.array(bc.get_boundary_values().values()) == 0.0):
				self.homogeneous = False
				break
		
		if self.homogeneous:
			self.bc_u_adj = self.bc_u
		else:
			self.bc_u_adj = []
			for bc in self.bc_u:
				r = bc.value().value_rank()
				self.bc_u_adj.append(DirichletBC(bc.function_space(), Constant([0]*problem.mesh_gdim) if r > 0 else Constant(0), bc.user_sub_domain(), bc.method()))
		
		# init functions
		self.f = self.getBoundaryTraction(problem.V_f)
		self.ps = self.getPointSources(problem.V_f)
		self.g0 = self.getVolumeForce(problem.V_g)
		self.g1 = self.getVolumeGravity(problem.V_g)
		self.u = Function(problem.V_u)
		self.u_adj = Function(problem.V_u)

		dx_ = problem.dx_
		ds_ = problem.ds_
		rho = problem.rho**problem.eta
		drho = problem.eta*(problem.rho**(problem.eta-1))

		# init forward problem
		self.mat_a = Matrix()
		self.vec_l = Vector()

		# objective and gradient
		phi = TestFunction(problem.V_rho)

		def C_phi(v, u1, u2):
			return 1/problem.eikonal_grad_norm*inner(problem.material_model.B(eps(u1), problem.eikonal_p, grad(v), problem.ascale, problem.iscale, problem.mesh_gdim), eps(u2))

		if app.parameters["objective_form"] == "compliance":
			# correct objective form
			pen = Constant(1.0)
			self.form_F = self.weight*problem.rho**pen*inner(problem.sigma(eps(self.u)), eps(self.u))*dx_
			self.form_F_rho = lambda v: self.weight*v*pen*problem.rho**(pen-1)*inner(problem.sigma(eps(self.u)), eps(self.u))*dx_
			self.form_F_u = lambda v: self.weight*problem.rho**pen*(inner(problem.sigma(eps(v)), eps(self.u)) + inner(problem.sigma(eps(self.u)), eps(v)))*dx_
			self.form_F_phi = lambda v: self.weight*problem.rho**pen*C_phi(v, self.u, self.u)*dx_
		elif app.parameters["objective_form"] == "penalized_compliance":
			# penalized objective form
			self.form_F = self.weight*inner(self.f, self.u)*ds_ + self.weight*inner(self.g0 + rho*self.g1, self.u)*dx_
			self.form_F_rho = lambda v: self.weight*inner(v*drho*self.g1, self.u)*dx_
			self.form_F_u = lambda v: self.weight*inner(self.f, v)*ds_ + self.weight*inner(self.g0 + rho*self.g1, v)*dx_
			self.form_F_phi = lambda v: self.weight*Constant(0)*v*dx_
			if self.homogeneous:
				self.u_adj = -self.u	# since self.form_F_u=self.form_L

		v = TestFunction(problem.V_u)
		self.form_L = inner(self.f, v)*ds_ + inner(self.g0 + rho*self.g1, v)*dx_
		self.form_L_rho = lambda v: inner(v*self.g1, self.u_adj)*dx_
		self.form_L_adj = -self.form_F_u(v)

		self.form_dF = self.form_F_rho(phi) + drho*phi*inner(problem.sigma(eps(self.u)), eps(self.u_adj))*dx_ - self.form_L_rho(phi)

		# init adjoint Eikonal linear form
		if isinstance(problem.material_model, TransverseIsotropicMaterialModel):

			v = TestFunction(problem.V_eikonal)

			self.form_eikonal_adj_l = -self.form_F_phi(v) - rho*C_phi(v, self.u, self.u_adj)*dx_

	def computeForward(self):

		problem = self.problem

		# solve elasticity forward problem
		assemble(self.form_L, self.vec_l)
		self.mat_a = problem.mat_a.copy()

		for bc in self.bc_u:
			bc.apply(self.mat_a)
			bc.apply(self.vec_l)

			if DEBUGX:
				print "bc_u:", bc.get_boundary_values()

		for ps in self.ps:
			ps.apply(self.vec_l)

		"""
		mumps         |  MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)
	petsc         |  PETSc built in LU solver                                    
	superlu_dist  |  Parallel SuperLU                                            
	umfpack
		"""

		self.solver_a = LUSolver(self.mat_a)
		self.solver_a.parameters["reuse_factorization"] = True
		self.solver_a.solve(self.u.vector(), self.vec_l)

		if DEBUGX:
			print "u:", self.u.vector().array()


	def computeAdjoint(self):
		
		if not isinstance(self.u_adj, Function):
			return
		
		# solve elasticity adjoint problem
		assemble(self.form_L_adj, self.vec_l)

		for bc in self.bc_u_adj:
			bc.apply(self.vec_l)
			if DEBUGX:
				print "bc_u_adj:", bc.get_boundary_values()

		self.solver_a.solve(self.u_adj.vector(), self.vec_l)

		if DEBUGX:
			print "u_adj:", self.u_adj.vector().array()


class TopologyOptimizationProblem(object):

	def __init__(self):
		app = QtGui.QApplication.instance()
		self.mesh_R = app.parameters["mesh_resolution"]*(app.parameters["initial_refinements"]+1)
		self.rho_bar = Constant(app.parameters["rho_bar"])
		self.tol = Constant(app.parameters["tol"])
		self.eta = Constant(app.parameters["eta"])
		self.gamma = Constant(app.parameters["gamma"])
		self.rho_min = Constant(app.parameters["rho_min"])
		self.ip_grad_scale = Constant(app.parameters["ip_grad_scale"])
		self.material_model = createInstance(app.parameters["material_model"], MaterialModel)
		self.eikonal_tau_scale = Constant(app.parameters["tau_scale"])
		self.eikonal_theta_scale = Constant(app.parameters["theta_scale"])
		self.eikonal_c_min = Constant(0.01)
		self.eps = 1e-6
		self.gradient_scaling = app.parameters["gradient_scaling"]
		self.renormalization = app.parameters["renormalization"]
		self.ascale = Constant(app.parameters["anisotropy_scale"])
		self.iscale = Constant(app.parameters["isotropy_degree"])
		self.alpha = Constant(app.parameters["alpha"])
		self.alpha_scale = Constant(app.parameters["alpha_scale"])
		self.penalty = Constant(app.parameters["penalty"])
		self.delta_rho = Constant(app.parameters["delta_rho"])
		self.rho0 = Constant(app.parameters["rho0"])
		self.num_forward = 0	# number of forward evaluations
		self.num_adjoint = 0	# number of adjoint evaluations
		self.alpha_relax = Constant(app.parameters["alpha_relax"])
		self.alpha_conv = Constant(app.parameters["alpha_conv"])
		self.dF_scale = None
		self.S = None

		self.setDensityFilename(app.parameters["density"])

		self.params = [
			Param(type="range", name="eta", getValue=lambda: float(self.eta), setValue=lambda x: self.eta.assign(x), min=0, max=10, digits=2),
			Param(type="range", name="rho_bar", getValue=lambda: float(self.rho_bar), setValue=lambda x: self.rho_bar.assign(x), min=0, max=1, digits=6),
			#Param(type="range", name="rho_min", getValue=lambda: float(self.rho_min), setValue=lambda x: self.rho_min.assign(x), min=0, max=1, digits=6),
			#Param(type="range", name="delta_rho", getValue=lambda: float(self.delta_rho), setValue=lambda x: self.delta_rho.assign(x), min=0, max=1, digits=6),
			#Param(type="range", name="rho0", getValue=lambda: float(self.rho0), setValue=lambda x: self.rho0.assign(x), min=0, max=1, digits=6),
			#Param(type="range", name="penalty", getValue=lambda: float(self.penalty), setValue=lambda x: self.penalty.assign(x), min=0, max=1, digits=6),
			Param(type="range", name="alpha", getValue=lambda: float(self.alpha), setValue=lambda x: self.alpha.assign(x), min=0.0, max=app.parameters["alpha_max"], digits=5),
			Param(type="range", name="alpha_relax", getValue=lambda: float(self.alpha_relax), setValue=lambda x: self.alpha_relax.assign(x), min=0.0, max=1.0, digits=4),
			#Param(type="range", name="alpha_conv", getValue=lambda: float(self.alpha_conv), setValue=lambda x: self.alpha_conv.assign(x), min=0.0, max=100000.0, digits=1),
		]

		if self.gradient_scaling == "mass_H1":
			self.params.append(Param(type="range", name="ip_grad_scale", getValue=lambda: float(self.ip_grad_scale), setValue=lambda x: self.ip_grad_scale.assign(x), min=0, max=10, digits=6))

		if isinstance(self.material_model, TransverseIsotropicMaterialModel):
			self.params.append(Param(type="range", name="ascale", getValue=lambda: float(self.ascale), setValue=lambda x: self.ascale.assign(x), min=0.0, max=1000.0, digits=2))
			self.params.append(Param(type="range", name="iscale", getValue=lambda: float(self.iscale), setValue=lambda x: self.iscale.assign(x), min=0.0, max=1.0, digits=6))

			if app.parameters["eikonal_solver"] == "internal":
				self.params.append(Param(type="range", name="tau_scale", getValue=lambda: float(self.eikonal_tau_scale), setValue=lambda x: self.eikonal_tau_scale.assign(x), min=0, max=10, digits=4))
			#self.params.append(Param(type="range", name="theta_scale", getValue=lambda: float(self.eikonal_theta_scale), setValue=lambda x: self.eikonal_theta_scale.assign(x), min=0, max=1, digits=4))
			#self.params.append(Param(type="range", name="c_min", getValue=lambda: float(self.eikonal_c_min), setValue=lambda x: self.eikonal_c_min.assign(x), min=0.0001, max=0.1, digits=4))
			pass

		self.interpolator = LagrangeInterpolator()
		#self.interpolator = FunctionExpression(degree=8)

	# return a list of loadcases
	def getLoadcases(self):
		return []

	# return mesh with resolution multiplier R (number of elements in one direction)
	def getMesh(self, R):
		raise NotImplementedError()

	# return initial material distribution
	def getInitialRho(self, V):
		return None

	# return DirichletBC for rho
	def getDirichletBC(self, V):
		return []

	# return injection domain
	def getInjectionDomain(self):
		return CompiledSubDomain("on_boundary")

	def interpolateFields(self, fields=None):

		if fields is None:
			fields = [(self.rho, self.rho_old)]
			if isinstance(self.material_model, TransverseIsotropicMaterialModel):
				fields += [(self.eikonal, self.eikonal_old)]

		for rho, rho_old in fields:

			if isinstance(self.interpolator, LagrangeInterpolator):
				self.interpolator.interpolate(rho, rho_old)
			else:
				self.interpolator.f = rho_old
				u = TrialFunction(self.V_rho)
				v = TestFunction(self.V_rho)
				solve(u*v*dx == self.interpolator*v*dx, rho)

	def refine(self, factor=2):
		self.mesh_R *= factor
		self.init_new_mesh()

	def adapt_cells(self, cell_markers):
		new_mesh = refine(self.mesh, cell_markers)
		self.init_new_mesh(new_mesh)

	def init_new_mesh(self, new_mesh=None):

		self.rho_old = self.rho
		rho_bar_old = assemble(self.rho_old*dx)/self.mesh_V

		self.eikonal_old = getattr(self, "eikonal", None)

		rho_filename = self.rho_filename
		self.setDensityFilename(None)
		getMeshOrg = self.getMesh
		if not new_mesh is None:
			self.getMesh = lambda res: new_mesh
		self.init()
		self.getMesh = getMeshOrg
		self.setDensityFilename(rho_filename)

		self.interpolateFields()
		self.projectDensity(rho_bar=rho_bar_old)

	# adaptive mesh refinement
	def kadapt(self):
		
		app = QtGui.QApplication.instance()
		cell_markers = CellFunction("bool", self.mesh)
		cell_markers.set_all(False)

		indicator = "rho"

		if indicator == "rho":
			rho_min = float(self.rho_min)
			tol = (rho_min+1)*0.5
			rho_dg0 = project(self.rho, self.V_DG0).vector().array()
			dofmap = self.V_DG0.dofmap()
			for cell in cells(self.mesh):
				if cell_markers[cell]:
					continue
				cellindex = cell.index()
				dofs = dofmap.cell_dofs(cellindex)
				dof = dofs[0]
				value = rho_dg0[dof]
				if value > tol:
					cell_markers[cell] = True
					# refine also the neighbour cells
					if 1:
						for ncell in cells(cell):
							cell_markers[ncell] = True
		elif indicator == "stress":
			# TODO: refinement based on normal jump of stress
			pass

		self.adapt_cells(cell_markers)  


	# adaptive mesh refinement
	def adapt(self):
		
		app = QtGui.QApplication.instance()
		cell_markers = CellFunction("bool", self.mesh)
		cell_markers.set_all(False)

		indicator = "rho"

		if indicator == "rho":
			rho_min = float(self.rho_min)
			tol = rho_min
			rho_dg0 = project(self.rho, self.V_DG0).vector().array()
			dofmap = self.V_DG0.dofmap()
			for cell in cells(self.mesh):
				if cell_markers[cell]:
					continue
				cellindex = cell.index()
				dofs = dofmap.cell_dofs(cellindex)
				dof = dofs[0]
				value = rho_dg0[dof]
				if (value - rho_min) > tol and (1.0 - value) > tol:
					cell_markers[cell] = True
					# refine also the neighbour cells
					if 1:
						for ncell in cells(cell):
							cell_markers[ncell] = True
		elif indicator == "stress":
			# TODO: refinement based on normal jump of stress
			pass

		self.adapt_cells(cell_markers)  

	# readaptive mesh refinement (finest cells are not refined)
	def readapt(self):
		
		hmin = float(self.mesh_hmin)
		rho_min = float(self.rho_min)
		tol = rho_min
		rho_dg0 = project(self.rho, self.V_DG0).vector().array()
		dofmap = self.V_DG0.dofmap()
		cell_markers = CellFunction("bool", self.mesh)
		cell_markers.set_all(False)
		for cell in cells(self.mesh):
			if cell_markers[cell]:
				continue
			h = cell.h()
			if h < 3*hmin:
				continue
			cellindex = cell.index()
			dofs = dofmap.cell_dofs(cellindex)
			dof = dofs[0]
			value = rho_dg0[dof]
			if (value - rho_min) > tol and (1.0 - value) > tol:
				cell_markers[cell] = True

		self.adapt_cells(cell_markers)  

	def init(self):

		app = QtGui.QApplication.instance()

		# init mesh
		if not self.mesh_filename is None:
			print "loading mesh", self.mesh_filename
			self.mesh = Mesh(self.mesh_filename)
		else:
			mesh = self.getMesh(self.mesh_R)
			if not isinstance(mesh, Mesh):
				# mesh the domain
				domain = mesh
				if domain.dim() == 2:
					# 2d
					#generator = mshr.CSGCGALMeshGenerator2D()
					#generator.parameters["mesh_resolution"] = float(self.mesh_R)
					#generator.parameters["triangle_shape_bound"] = 0.3
					#generator.parameters["cell_size"] = 1.0
					#generator.parameters["lloyd_optimize"] = False
					#generator.parameters["edge_truncate_tolerance"] = 1e-16
					#generator.parameters["partition"] = True
					#mesh = Mesh()
					#generator.generate(domain, mesh)
					mesh = mshr.generate_mesh(domain, self.mesh_R)
				else:
					# 3d
					generator = mshr.CSGCGALMeshGenerator3D()
					generator.parameters["mesh_resolution"] = float(self.mesh_R)
					generator.parameters["perturb_optimize"] = False
					generator.parameters["exude_optimize"] = False
					generator.parameters["lloyd_optimize"] = False
					generator.parameters["odt_optimize"] = False
					generator.parameters["edge_size"] = 0.025
					generator.parameters["facet_angle"] = 25.0
					generator.parameters["facet_size"] = 0.05
					generator.parameters["facet_distance"] = 0.005
					generator.parameters["cell_radius_edge_ratio"] = 3.0
					generator.parameters["cell_size"] = 0.25
					generator.parameters["detect_sharp_features"] = True
					generator.parameters["feature_threshold"] = 70.0
					mesh = generator.generate(domain)
			self.mesh = mesh

		self.mesh_gdim = self.mesh.geometry().dim()
		self.mesh_V = assemble(Constant(1)*dx(self.mesh))
		self.mesh_hmax = self.mesh.hmax()
		self.mesh_hmin = self.mesh.hmin()

		# init function spaces
		self.V_DG0 = FunctionSpace(self.mesh, "DG", 0)
		self.V_P1 = FunctionSpace(self.mesh, "Lagrange", 1)
		self.V_A2 = TensorFunctionSpace(self.mesh, "DG", 0)

		u_space = app.parameters["u_space"]
		if u_space == "CG1":
			self.V_u = VectorFunctionSpace(self.mesh, "Lagrange", 1)
		elif u_space == "CG2":
			self.V_u = VectorFunctionSpace(self.mesh, "Lagrange", 2)
		else:
			print u_space
			raise NotImplementedError()


		rho_space = app.parameters["rho_space"]
		if rho_space == "CG1":
			self.V_rho = self.V_P1
			self.V_f = self.V_g = self.V_u
		elif rho_space == "DG0":
			self.V_rho = self.V_DG0
			self.V_f = self.V_g = VectorFunctionSpace(self.mesh, "DG", 0)
		else:
			print rho_space
			raise NotImplementedError()

		
		if isinstance(self.material_model, FileTransverseIsotropicMaterialModel):
			vtk_file = app.parameters["fo_file"]
			self.A2 = interpolate(FO_Expr(vtk_file), self.V_A2)

		# init functions
		rho_init = app.parameters["rho_init"]
		self.rho = None
		if rho_init == 'auto' or not self.rho_filename is None:
			if not self.rho_filename is None:
				print "loading rho", self.rho_filename
				self.rho = Function(self.V_rho, self.rho_filename)
				self.rho_bar.assign(round(assemble(self.rho*dx)/self.mesh_V, 5))
			elif not app.parameters["algorithm"] in ["GuideWeight"]:
				self.rho = self.getInitialRho(self.V_rho)
				if not self.rho is None:
					self.rho_bar.assign(assemble(self.rho*dx)/self.mesh_V)

		if self.rho is None:
			self.rho = Function(self.V_rho)

			if rho_init in ['constant', 'auto']:
				rho0 = float(self.rho0)
				rhobar = float(self.rho_bar)
				rhoinit = rho0 if rho0 > 0.0 else rhobar
				print "set constant rho", rhoinit
				self.rho.vector()[:] = rhoinit
			elif rho_init == 'random':
				print "set random rho"
				n = self.rho.vector().size()
				#alpha = (1 - float(self.rho_min))/(float(self.rho_bar) - float(self.rho_min)) - 1
				#self.rho.vector()[:] = float(self.rho_min) + (np.random.rand(n)**alpha)*(1 - float(self.rho_min))
				self.rho.vector()[:] = np.random.rand(n)*(2*float(self.rho_bar) - float(self.rho_min))
			else:
				print rho_init
				raise NotImplementedError()

		renormalize = float(self.rho_bar)

		self.rho_temp = Function(self.V_rho)

		self.bc_rho = self.getDirichletBC(self.V_rho)
		self.bc_rho0 = []
		self.bc_rho1 = []
		for bc in self.bc_rho:
			r = bc.value().value_rank()
			self.bc_rho0.append(DirichletBC(bc.function_space(), Constant(0), bc.user_sub_domain(), bc.method()))
			self.bc_rho1.append(DirichletBC(bc.function_space(), Constant(1), bc.user_sub_domain(), bc.method()))

		self.rho_bc_ind = Function(self.V_rho)
		self.rho_bc_ind.vector().zero()
		for bc in self.bc_rho1:
			bc.apply(self.rho_bc_ind.vector())
		
		for bc in self.bc_rho:
			bc.apply(self.rho.vector())

		self.dF = Function(self.V_rho)
		self.dF_temp = Function(self.V_rho)
		self.dF_temp_P1 = Function(self.V_P1)
		self.dF_weights = Function(self.V_rho)
		self.dF_scale = None

		if isinstance(self.material_model, TransverseIsotropicMaterialModel):
			self.V_eikonal = self.V_P1
			self.P_eikonal = VectorFunctionSpace(self.mesh, "DG", 0)

			idom = CompiledSubDomain(app.parameters["injection_domain"]) if app.parameters["injection_domain"] else self.getInjectionDomain()

			if bool(app.parameters["eikonal_dual"]):
				# compute dual of idom
				idom_org = idom
				class DualInjectionDomain(SubDomain):
					def inside(self, x, on_boundary):
						return on_boundary and not idom_org.inside(x, on_boundary)
				idom = DualInjectionDomain()
				self.eikonal_theta = self.eta*self.eikonal_theta_scale
			else:
				# self.eikonal_theta = 0.5*(1 - self.eta)*self.eikonal_theta_scale
				self.eikonal_theta = -self.eta*self.eikonal_theta_scale

			self.eikonal_bc = DirichletBC(self.V_eikonal, Constant(0), idom, method=app.parameters["injection_domain_method"])
			self.eikonal = Function(self.V_eikonal)
			self.eikonal_adj = Function(self.V_eikonal)
			self.eikonal_p = Function(self.P_eikonal)

			if app.parameters["eikonal_solver"] == "external" and self.mesh_gdim == 2:
				self.eikonal_grad_norm = self.rho**self.eikonal_theta
			else:
				self.eikonal_grad_norm = sqrt(inner(grad(self.eikonal), grad(self.eikonal))) # + self.eps

		sig = self.V_rho.dolfin_element().signature()
		if sig == "FiniteElement('Lagrange', triangle, 1)" or \
		   sig == "FiniteElement('Lagrange', tetrahedron, 1)" or \
		   sig == "FiniteElement('Lagrange', Domain(Cell('triangle', 2)), 1, None)" or \
		   sig == "FiniteElement('Lagrange', Domain(Cell('tetrahedron', 3)), 1, None)":
			# TODO does not work in parallel
			if app.mpi_rank == 1:
				v2d = vertex_to_dof_map(self.V_rho)
				for v in entities(self.mesh, 0):
					volume = 0.0
					for c in entities(v, self.mesh_gdim):
						volume += Cell(self.mesh, c.index()).volume()
					dof_index = v2d[v.index()]
					self.dF_weights.vector()[dof_index] = 1/volume
		elif sig == "FiniteElement('Discontinuous Lagrange', triangle, 0)" or \
		     sig == "FiniteElement('Discontinuous Lagrange', Domain(Cell('triangle', 2)), 0, None)":
			assemble(Constant(1)*TestFunction(self.V_rho)*dx, tensor=self.dF_weights.vector())
			self.dF_weights.vector()[:] = 1/self.dF_weights.vector().array()
		else:
			print sig
			raise NotImplementedError()

		# create measures
		# metadata={'quadrature_degree': 1, 'quadrature_rule': 'default', 'optimize': True})
		self.dx_ = dx_ = dx # (degree=1)
		self.ds_ = ds_ = ds

		# init forms
		u = TrialFunction(self.V_u)
		v = TestFunction(self.V_u)
		phi = TestFunction(self.V_rho)

		if isinstance(self.material_model, TransverseIsotropicMaterialModel):
			self.sigma = lambda eps: self.material_model.sigma(eps, self.eikonal_p, self.ascale, self.iscale, self.mesh_gdim)
		elif isinstance(self.material_model, FileTransverseIsotropicMaterialModel):
			self.sigma = lambda eps: self.material_model.sigma(eps, self.A2, self.ascale, self.iscale, self.mesh_gdim)
		else:
			self.sigma = lambda eps: self.material_model.sigma(eps, self.mesh_gdim)

		rho = self.rho**self.eta
		drho = self.eta*(self.rho**(self.eta-1))

		# forward problem
		self.form_a = inner(rho*self.sigma(eps(u)), eps(v))*dx_

		# init loadcases, objective and gradient
		self.loadcases = self.getLoadcases()
		self.form_F = None
		self.form_dF = None
		for lc in self.loadcases:
			self.form_F = lc.form_F if self.form_F is None else (self.form_F + lc.form_F)
			if app.parameters["gradient_terms"] in ["all", "isotropic"]:
				self.form_dF = lc.form_dF if self.form_dF is None else (self.form_dF + lc.form_dF)

		if isinstance(self.material_model, TransverseIsotropicMaterialModel):

			self.form_eikonal_adj_l = None
			for lc in self.loadcases:
				self.form_eikonal_adj_l = lc.form_eikonal_adj_l if self.form_eikonal_adj_l is None else (self.form_eikonal_adj_l + lc.form_eikonal_adj_l)

			# add contribution to gradient
			if app.parameters["gradient_terms"] in ["all", "anisotropic"]:
				form_dF = -phi*self.eikonal_theta*(self.rho**(self.eikonal_theta-1))*self.eikonal_adj*dx_
				self.form_dF = form_dF if self.form_dF is None else (self.form_dF + form_dF)


		"""
		class MyExpression0(Expression):
			def __init__(self, dF, rho, rho_min, **kwargs):
				self.dF = dF
				self.rho = rho
				self.rho_min = rho_min
			def eval_cell(self, value, x, ufc_cell):
				v = np.zeros(1, dtype=np.double)
				self.rho.eval_cell(v, x, ufc_cell)
				rho = v[0]
				self.dF.eval_cell(v, x, ufc_cell)
				dF = v[0]
				eps = 1e-3
				if (dF > 0 and rho <= self.rho_min + eps):
					value[0] = 0.0
				elif (dF < 0 and rho >= 1.0 - eps):
					value[0] = 0.0
				else:
					value[0] = dF

		#self.form_correct_dF = phi*self.dF*conditional(ge(self.dF, 0), self.rho**self.eta - self.rho_min, 1 - self.rho**self.eta)*dx_
		
		self.form_correct_dF = MyExpression0(self.dF, self.rho, float(self.rho_min), degree=1)
		"""

		self.rho_gamma_tau = self.eps
		self.rho_gamma_upsilon = 2*self.rho_gamma_tau
		xi = lambda rho: (rho - self.rho_min + self.rho_gamma_tau)/(1 - self.rho_min + self.rho_gamma_upsilon)
		gamma = self.gamma**(1 - self.rho_bc_ind)
		self.form_int_rho_gamma = (self.rho_min - self.rho_gamma_tau + (1-self.rho_min + self.rho_gamma_upsilon)*xi(self.rho)**gamma)*dx_
		self.form_int_rho_gamma_d = ((1-self.rho_min + self.rho_gamma_upsilon)*ln(xi(self.rho)**(xi(self.rho)**gamma)))*dx_

		self.form_int_dF = self.dF*dx_
		self.form_int_abs_dF = abs(self.dF)*dx_
		self.scale_0 = None

		u = TrialFunction(self.V_P1)
		v = TestFunction(self.V_P1)

		def proj(expr, a, b):
			return conditional(gt(expr, b), b, conditional(lt(expr, a), a, expr))

		self.beta = Constant(0.0)
		self.expr_gradient_step = self.rho - self.alpha*self.dF + self.beta*(1 - self.rho_bc_ind)
		self.expr_projected_gradient = self.rho_bar - proj(self.expr_gradient_step, self.rho_min, 1.0)
		self.form_int_proj_dF = self.expr_projected_gradient*dx_

		self.form_dF_mass_L2 = inner(u,v)*dx_
		self.mat_dF_mass_L2 = assemble(self.form_dF_mass_L2)
		self.dF_mass_solver_L2 = LUSolver()
		self.dF_mass_solver_L2.parameters['reuse_factorization'] = True
		self.dF_mass_solver_L2.set_operator(self.mat_dF_mass_L2)


		#h = Circumradius(self.mesh)
		h = Constant(0.5*(self.mesh_hmin + self.mesh_hmax))
		print "h=", float(h)
		h2 = h**2
		self.form_dF_mass_H1 = (inner(u,v) + self.ip_grad_scale*h2*inner(grad(u), grad(v)))*dx_
		self.mat_dF_mass_H1 = assemble(self.form_dF_mass_H1)
		self.dF_mass_solver_H1 = LUSolver()
		self.dF_mass_solver_H1.parameters['reuse_factorization'] = True
		self.dF_mass_solver_H1.set_operator(self.mat_dF_mass_H1)

		# init Eikonal solver
		if isinstance(self.material_model, TransverseIsotropicMaterialModel):
			u = TrialFunction(self.V_eikonal)
			v = TestFunction(self.V_eikonal)
			#self.eikonal_tau = Constant(1.0*self.mesh_hmin)*self.eikonal_tau_scale
			# TODO: should we really scale by h?
			self.eikonal_tau = h*self.eikonal_tau_scale

			if app.parameters["eikonal_solver"] == "external" and self.mesh_gdim == 2:
				# use as preconditioner (initial guess for Newton solver)
				self.eikonal2d_solver = eikonal2d.Eikonal2d()
				self.eikonal2d_solver.set_epsilon(1e-12)
				points = self.mesh.coordinates().ravel(order='C')
				cells = self.mesh.cells().ravel(order='C')
				self.eikonal2d_solver.init_mesh(points.tolist(), cells.tolist())
				bc = self.eikonal_bc.get_boundary_values()
				self.eikonal_d2v = dof_to_vertex_map(self.V_eikonal)
				self.eikonal_v2d = vertex_to_dof_map(self.V_eikonal)
				seed_points = [int(self.eikonal_d2v[dof]) for dof, value in bc.iteritems()]
				seed_values = [value for dof, value in bc.iteritems()]
				self.eikonal2d_solver.set_seed_points(seed_points, seed_values)
				# print points.tolist(), cells.tolist(), seed_points, seed_values
				# TODO
				self.eikonal_tau = Constant(0.0)
			else:
				self.form_eikonal_init_a = self.eikonal_tau*inner(grad(u), grad(v))*dx_
				self.form_eikonal_init_L = (self.rho**self.eikonal_theta)*v*dx_

				self.form_eikonal_F = (self.eikonal_grad_norm*v + self.eikonal_tau*inner(grad(self.eikonal), grad(v)) - (self.rho**self.eikonal_theta)*v)*dx_
				self.form_eikonal_J = derivative(self.form_eikonal_F, self.eikonal, u)

				problem = NonlinearVariationalProblem(self.form_eikonal_F, self.eikonal, self.eikonal_bc, J=self.form_eikonal_J)
				self.eikonal_solver = NonlinearVariationalSolver(problem)

				prm = self.eikonal_solver.parameters
				prm['newton_solver']['absolute_tolerance'] = 1e-14
				prm['newton_solver']['relative_tolerance'] = 1e-12
				prm['newton_solver']['maximum_iterations'] = 25
				prm['newton_solver']['relaxation_parameter'] = 1.0
				prm['newton_solver']['report'] = False
				prm['newton_solver']['report'] = True
				#prm['newton_solver']['linear_solver'] = "cg"
				#prm['newton_solver']['preconditioner'] = "amg"


				"""
				  p.add("linear_solver",           "default");
				  p.add("preconditioner",          "default");
				  p.add("convergence_criterion",   "residual");
				  p.add("method",                  "full");
				  p.add("error_on_nonconvergence", true);
				"""

			self.form_eikonal_adj_a = inner(self.eikonal_p*u + self.eikonal_tau*grad(u), grad(v))*dx_

			self.mat_eikonal_adj_a = Matrix()
			self.vec_eikonal_adj_l = Vector()
			
			self.eikonal_adj_solver = LUSolver()
			self.eikonal_adj_solver.parameters['reuse_factorization'] = False

			u = TrialFunction(self.P_eikonal)
			v = TestFunction(self.P_eikonal)
			self.form_eikonal_p_a = inner(u, v)*dx_

			if bool(app.parameters["eikonal_dual"]):
				# rotate fibers in void region by 90 degree
				phi = 0.5*np.pi*(1 - self.rho/(1-self.rho_min))
				phi = 0.5*np.pi
				R = as_matrix([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])
				self.form_eikonal_p_l = inner(dot(R, grad(self.eikonal)), v)/self.eikonal_grad_norm*dx_
			else:
				self.form_eikonal_p_l = inner(grad(self.eikonal), v)/self.eikonal_grad_norm*dx_

			self.mat_eikonal_p_a = assemble(self.form_eikonal_p_a)
			self.vec_eikonal_p_l = Vector()
			self.eikonal_p_solver = LUSolver()
			self.eikonal_p_solver.parameters['reuse_factorization'] = True
			self.eikonal_p_solver.set_operator(self.mat_eikonal_p_a)


		# init forward problem
		self.mat_a = Matrix()

		# reset smoothing operator
		self.S = None

		if renormalize:
			print "renormalize", renormalize
			self.projectDensity(rho_bar=renormalize)
			print "rho_bar=", assemble(self.rho*dx)/self.mesh_V


	def buildSmoothingOperator(self):

		if not self.S is None:
			return

		a1 = self.mesh.cells()
		a1.flags.writeable = False
		a2 = self.mesh.coordinates()
		a2.flags.writeable = False
		m = hashlib.md5()
		m.update(a1.data)
		m.update(a2.data)
		mesh_hash = m.hexdigest()
		S_filename = ".smooth_%s.npz" % mesh_hash
		print "S-operator filename:", S_filename

		def save_sparse_csr(filename, array):
			np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)

		def load_sparse_csr(filename):
			loader = np.load(filename)
			return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

		if os.path.isfile(S_filename):
			self.S = load_sparse_csr(S_filename)
			return

		def _getNeighbourCells(cell, levels, cell_list):
			# slow version
			cell_list.add(cell.index())
			if levels > 0:
				levels -= 1
				for vertex in vertices(cell):
					for ncell in cells(vertex):
						getNeighbourCells(ncell, levels, cell_list)

		def getNeighbourCells(mesh, cell, levels):

			new_cells = [cell]
			new_cells_index = set([cell.index()])
			#visited_vertices = []
			visited_vertices_index = set()
			#visited_cells = []
			visited_cells_index = set()
			
			for level in range(levels):
				new_vertices = []
				new_vertices_index = set()
				for cell in new_cells:
					for vertex in vertices(cell):
						if vertex.index() in visited_vertices_index:
							continue
						new_vertices_index.add(vertex.index())
						new_vertices.append(vertex)
				#visited_cells.extend(new_cells)
				visited_cells_index.update(new_cells_index)
				new_cells = []
				new_cells_index = set()
				for vertex in new_vertices:
					for cell in cells(vertex):
						if cell.index() in visited_cells_index:
							continue
						new_cells_index.add(cell.index())
						new_cells.append(cell)
				visited_vertices_index.update(new_vertices_index)

			visited_cells_index.update(new_cells_index)
			#visited_cells.extend(new_cells)

			return visited_cells_index

		# compute hash for operator
		self.mesh.cells()
		self.mesh.coordinates()

		# build smoothing operator S
		smooth_levels = 1
		num_cells = self.V_DG0.dim()
		S = scipy.sparse.lil_matrix((num_cells, num_cells))
		dofmap = self.V_DG0.dofmap()
		dofs = np.zeros(num_cells, dtype=np.int)
		for i in range(num_cells):
			dofs[i] = dofmap.cell_dofs(i)[0]
		for cell in cells(self.mesh):
			cell_index = cell.index()
			if cell_index % 100 == 0:
				sys.stdout.write("\rBulding smoothing operator %d%%" % (cell_index*100/num_cells))
			row = dofs[cell_index]
			ncells = list(getNeighbourCells(self.mesh, cell, smooth_levels))
			ncells.sort()
			smooth_radius = (smooth_levels + 2)*cell.circumradius()
			cellmp = cell.midpoint()
			for ncell_index in ncells:
				col = dofs[ncell_index]
				ncell = Cell(self.mesh, ncell_index)
				d = cellmp.distance(ncell.midpoint())
				#S[row, col] = ncell.volume()*max(0.0, 1.0 - d/smooth_radius)
				S[row, col] = 1 if ncell_index == cell_index else 1.0/3.0

		S = scipy.sparse.csr_matrix(S)
		# scale rows
		v = np.array(S.sum(axis=0)).ravel()
		self.S = scipy.sparse.spdiags(1.0/v, 0, len(v), len(v))*S

		# save to cache
		save_sparse_csr(S_filename, self.S)

		sys.stdout.write("\n")

	def computeForward(self):

		print "compute forward"

		if DEBUGX:
			print "rho:", self.rho.vector().array()

		self.num_forward += 1

		##########################################################
		# Eikonal equation
		if isinstance(self.material_model, TransverseIsotropicMaterialModel):
			
			if hasattr(self, "eikonal2d_solver"):
				# TODO: assuming constant element speed here, but is not
				v = assemble(self.rho**-self.eikonal_theta*TestFunction(self.V_DG0)/CellVolume(self.mesh)*dx).array()
				self.eikonal2d_solver.set_element_speed(v.tolist())
				#print "minx/max/", v.min(), v.max()
				#self.eikonal2d_solver.set_element_speed([1.0]*len(v))
				t = self.eikonal2d_solver.compute_time()
				self.eikonal.vector()[:] = np.array(t)[self.eikonal_d2v]
				#self.eikonal_solver.solve()
			else:
				# use Newton solver

				# initial guess for solution of eikonal equation
				if not np.any(self.eikonal.vector().array()):
					solve(self.form_eikonal_init_a == self.form_eikonal_init_L, self.eikonal, self.eikonal_bc)

				# solve eikonal equation
				# solve(self.form_eikonal_F == 0, self.eikonal, self.eikonal_bc, J=self.form_eikonal_J)
				self.eikonal_solver.solve()

			if DEBUGX:
				print "eikonal:", self.eikonal.vector().array()

			# compute fiber orientation
			# solve(self.form_eikonal_p_a == self.form_eikonal_p_l, self.eikonal_p)
			assemble(self.form_eikonal_p_l, self.vec_eikonal_p_l)
			self.eikonal_p_solver.solve(self.eikonal_p.vector(), self.vec_eikonal_p_l)
			
			if DEBUGX:
				print "fo:", self.eikonal_p.vector().array()


		##########################################################
		# solve elasticity problems
		assemble(self.form_a, self.mat_a)
		for lc in self.loadcases:
			lc.computeForward()

	def computeAdjoint(self):

		self.num_adjoint += 1

		app = QtGui.QApplication.instance()

		##########################################################
		# compute adjoint eikonal equation
		# solve(self.form_eikonal_adj_a == self.form_eikonal_adj_l, self.eikonal_adj, self.eikonal_bc)
		if isinstance(self.material_model, TransverseIsotropicMaterialModel) and app.parameters["gradient_terms"] in ["all", "anisotropic"]:

			assemble(self.form_eikonal_adj_a, self.mat_eikonal_adj_a, finalize_tensor=True)
			assemble(self.form_eikonal_adj_l, self.vec_eikonal_adj_l)
			
			"""
			for i in range(self.mat_eikonal_adj_a.size(0)):
				indices, values = self.mat_eikonal_adj_a.getrow(i)
				for j in range(len(indices)):
					if indices[j] == i:
						continue
					if values[j] < 0:
						values[j] = 0
				self.mat_eikonal_adj_a.setrow(i, indices, values)
				self.mat_eikonal_adj_a.apply("insert")
			"""

			self.eikonal_bc.apply(self.mat_eikonal_adj_a)
			self.eikonal_bc.apply(self.vec_eikonal_adj_l)

			self.eikonal_adj_solver.set_operator(self.mat_eikonal_adj_a)
			self.eikonal_adj_solver.solve(self.eikonal_adj.vector(), self.vec_eikonal_adj_l)

			if DEBUGX:
				print "eikonal_adj:", self.eikonal_adj.vector().array()

			#plot(self.eikonal, interactive=True)
			#plot(self.eikonal_p, interactive=True)
			#plot(self.eikonal_adj, interactive=True)
			#return

		##########################################################
		# solve adjoint elasticity problems
		for lc in self.loadcases:
			lc.computeAdjoint()

	def computeGradient(self, rescale=True, rebalance=False, forward=True):

		if (forward):
			self.computeForward()

		print "compute gradient"

		self.computeAdjoint()

		# assemble gradient
		assemble(self.form_dF, self.dF.vector())

		if DEBUGX:
			print "dF:", self.dF.vector().array()

		if rescale:
			# compute Riesz of derivative dF
			print "rescaling gradient"

			if self.gradient_scaling == "lumped":
				# use lumped mass (volume of support)
				self.dF.vector()[:] = self.dF.vector().array()*self.dF_weights.vector().array()
			elif self.gradient_scaling == "mass_L2":
				# use inverse mass matrix
				self.dF_temp.assign(self.dF)
				self.dF_mass_solver_L2.solve(self.dF.vector(), self.dF_temp.vector())
			elif self.gradient_scaling == "mass_H1":
				# use inverse mass matrix
				self.dF_temp.assign(self.dF)
				self.dF_mass_solver_H1.solve(self.dF.vector(), self.dF_temp.vector())
			elif self.gradient_scaling == "smooth_DG":
				# smoothing
				#self.dF.vector()[:] = self.dF.vector().array()*self.dF_weights.vector().array()
				f = project(self.dF, self.V_P1)
				self.dF_mass_solver_H1.solve(self.dF_temp_P1.vector(), f.vector())
				f = project(self.dF_temp_P1, self.V_DG0)
				self.dF.vector()[:] = f.vector()[:]
			elif self.gradient_scaling == "smooth":
				# smoothing
				self.buildSmoothingOperator()
				self.dF.vector()[:] = self.dF.vector().array()*self.dF_weights.vector().array()
				self.dF.vector()[:] = self.S.dot(self.dF.vector().array())
			elif self.gradient_scaling == "none":
				# no scaling
				pass
			else:
				print self.gradient_scaling
				raise NotImplementedError()
	

		"""
			# TODO: REMOVE		
			return

			if rebalance:
				
				self.dF.vector()[:] -= assemble(self.form_int_dF)/self.mesh_V

				# apply DirichletBC
				for bc in self.bc_rho0:
					bc.apply(self.dF.vector())

				# TODO: uncomment
				# self.dF.vector()[:] /= assemble(self.form_int_abs_dF)/self.mesh_V + self.eps

				#if self.scale_0 is None:
				self.scale_0 = 1.0/(assemble(self.form_int_abs_dF)/self.mesh_V + self.eps)
				self.dF.vector()[:] *= self.scale_0

				if self.renormalization == "additive":
					 self.bisectionMethod()
				elif self.renormalization == "scaling":
					 self.nonlinearScalingMethod()
				else:
					print self.renormalization
					raise NotImplementedError()
		"""


	def checkAdjoint(self, num_indices=0):

		rho = self.rho.vector().array()

		has_eikonal = hasattr(self, "eikonal_bc")

		if has_eikonal:
			bc_indices = self.eikonal_bc.get_boundary_values().keys()
		else:
			bc_indices = []

		non_bc_indices = list(set(range(len(rho))) - set(bc_indices))
		indices = []

		"""
		# check some boundary nodes
		if len(bc_indices) > 0:
			imax = np.argmax(rho[bc_indices])
			indices.append(bc_indices[imax])
			imin = np.argmin(rho[bc_indices])
			indices.append(bc_indices[imin])

		# check some non-boundary nodes
		if len(non_bc_indices) > 0:
			imax = np.argmax(rho[non_bc_indices])
			indices.append(non_bc_indices[imax])
			imin = np.argmin(rho[non_bc_indices])
			indices.append(non_bc_indices[imin])
		"""

		# check some nodes with maximal eikonal adjoint
		# TODO: does not work if space for self.eikonal_adj and rho space are different
		"""
		if hasattr(self, "eikonal_adj"):
			rem_indices = list(set(range(len(rho))) - set(indices))
			if len(rem_indices) > 0:
				imax = np.argmax(np.abs(self.eikonal_adj.vector().array()[rem_indices]))
				indices.append(rem_indices[imax])
		"""


		if has_eikonal:
			self.eikonal.vector()[:] = 0.0
		self.computeGradient(False)
		dF_exact = np.copy(self.dF.vector().array())
		F0 = self.computeObjective()

		rem_indices = list(set(range(len(rho))) - set(indices))
		if len(rem_indices) > 0:
			imax = np.argmax(np.abs(dF_exact[rem_indices]))
			indices.append(rem_indices[imax])

		# use all indices if only few
		if len(rho) <= 4:
			indices = range(len(rho))

		if num_indices > 0:
			indices = indices[0:num_indices]

		dF_list = []
		dF_fd_list = []
		h_list = []
		for index in indices:

			rho_old = float(self.rho.vector()[index])

			print "#"*80
			print "checking node %d, rho=%g, dF=%g" % (index, rho[index], dF_exact[index])

			#for i in [3,4,5,6,7]:
			for i in range(14):
				
				delta = 10**-i

				self.rho.vector()[index] = rho_old + delta
				if has_eikonal:
					self.eikonal.vector()[:] = 0.0
				self.computeForward()
				F1 = self.computeObjective()

				dF_fd = (F1 - F0)/delta
				
				#print "F0 = %g, F1 = %g" % (F0, F1)
				print "# delta=%g, dF_exact = %g, dF_fd = %g" % (delta, dF_exact[index], dF_fd)

				dF_list.append(dF_exact[index])
				dF_fd_list.append(dF_fd)
				h_list.append(delta)

			self.rho.vector()[index] = rho_old

		return h_list, dF_list, dF_fd_list

	def projectDensity(self, rho_bar=None, eps=None):

		rho_bar = float(self.rho_bar) if rho_bar is None else rho_bar
		rho_min = float(self.rho_min)
		gamma = float(self.penalty)
		rho = self.rho.vector().array()
		rho_new = self.rho_temp
		tau_plus = 0.5*gamma*(1 + rho_min)

		# determine Lagrange multiplier by bisection

		def func(lmbda):
			rho_new.vector()[:] = np.minimum(np.maximum(rho_min, 1.0/(1.0 - gamma)*(rho - lmbda - tau_plus)), 1.0)
			delta_rho = rho_bar - assemble(rho_new*self.dx_)/self.mesh_V
			return delta_rho

		if eps is None:
			eps = 1e-8 # float(self.tol)*rho_min

		a = np.min(rho) - (1.0 - gamma)*rho_bar - tau_plus
		b = np.max(rho) - (1.0 - gamma)*rho_bar - tau_plus

		lmbda = bisect(func, a, b, rtol=eps)

		# tweak to change the equality <rho> = rho_bar to the inequality <rho> <= rho_bar
		# this hardly happens for "normal" problems
		# NOTE: disabled, since this causes trouble, due to numerical errors the <rho> does decrease
		#if c <= 0.0:
		#	c = 0.0
		#	fc = func(c)

		# assign projected solution
		self.rho.vector()[:] = rho_new.vector()[:]


	"""
	def bisectionMethod(self):

		# Bisection method

		rho_next = self.rho.vector().array() - float(self.alpha)*self.dF.vector().array()
		a = float(self.rho_bar) - np.max(rho_next)  # = beta_plus > 0
		b = float(self.rho_bar) - np.min(rho_next)  # = beta_minus < 0

		self.beta.assign(a)
		fa = assemble(self.form_int_proj_dF)/self.mesh_V
		self.beta.assign(b)
		fb = assemble(self.form_int_proj_dF)/self.mesh_V

		for i in range(32):

			# fa + (c - a)*(fb - fa)/(b - a) = 0
			c = 0.5*(a+b)
			#c = a - fa*(b - a)/(fb - fa)
			self.beta.assign(c)
			fc = assemble(self.form_int_proj_dF)/self.mesh_V

			if abs(b-a) < self.eps or abs(fc) < self.eps:
				break

			print "bisection beta=%g, f=%g" % (c, fc)

			if (fc < 0):
				b = c
				fb = fc
			else:
				a = c
				fa = fc

		# correct gradient
		self.dF.vector()[:] = self.rho.vector().array() - np.minimum(np.maximum(float(self.rho_min), rho_next + float(self.beta)*(1 - self.rho_bc_ind.vector().array())), 1.0)

		print "rho_bar=", assemble(self.rho*dx)/self.mesh_V


	def nonlinearScalingMethod(self):
		
		rho_old = self.rho.vector().array()
		self.rho.vector()[:] = np.minimum(np.maximum(float(self.rho_min), rho_old - float(self.alpha)*self.dF.vector().array()), 1.0)
		self.renormalizeRho()
		
		print "done"
		# correct gradient
		self.dF.vector()[:] = rho_old - self.rho.vector().array()
		self.rho.vector()[:] = rho_old
		print "done"

	def renormalizeRho(self):

		rho = self.rho.vector()
		rho_min = float(self.rho_min)

		# project to [rho_min,1]
		rho[:] = np.maximum(rho_min, np.minimum(rho.array(), 1.0))
		
		print "rho_min=%g, rho_max=%g" % (np.min(rho.array()), np.max(rho.array()))

		#if isinstance(self, ParameterIdentificationTopologyOptimizationProblem):
		#	return

		self.gamma.assign(1.0)
		maxiter = 10
		for i in range(maxiter):
			r = assemble(self.form_int_rho_gamma)/self.mesh_V - float(self.rho_bar)
			if abs(r) < self.eps:
				print "renormalize rho gamma=%g" % float(self.gamma)
				break
			q = assemble(self.form_int_rho_gamma_d)/self.mesh_V
			self.gamma.assign(max(DOLFIN_EPS, float(self.gamma) - r/q))
			print "renormalize rho %d, r=%g, q=%g, r/q=%g, gamma=%g" % (i, r, q, r/q, float(self.gamma))

		rho_gamma_tau = float(self.eps)
		rho_gamma_upsilon = 2*rho_gamma_tau
		xi = lambda rho: (rho - rho_min + rho_gamma_tau)/(1 - rho_min + rho_gamma_upsilon)
		rho[:] = rho_min - rho_gamma_tau + (1-rho_min + rho_gamma_upsilon)*xi(rho.array())**(float(self.gamma)**(1 - self.rho_bc_ind.vector().array()))
	"""

	def computeObjective(self):
		return assemble(self.form_F)

	def setDensityFilename(self, filename):
		if filename is None or filename == "":
			self.mesh_filename = None
			self.rho_filename = None
			return
		filename_base, ext = os.path.splitext(filename)
		self.mesh_filename = filename_base + "_mesh.xml"
		self.rho_filename = filename


class ComplianceTopologyOptimizationProblem(TopologyOptimizationProblem):
	pass


class OptimalityCriterion(object):
	def eval(self, I, derivative=False):
		pass


class AOptimalityCriterion(OptimalityCriterion):
	def eval(self, I, derivative=False):


		Iinv = np.linalg.inv(I)
		trIinv = np.trace(Iinv)
		Psi = np.log(trIinv)

		#Psi = np.trace(I)
		if not derivative:
			return Psi
		dPsidI = (-1.0/trIinv)*np.dot(Iinv, Iinv)

		#dPsidI = np.identity(2)

		"""
		print "#", dPsidI
		i = 0
		j = 1
		Eij = np.zeros_like(I)
		Eij[i,j] = 1
		print 1.0/trIinv*np.trace(-np.dot(Iinv, np.dot(Eij, Iinv)))
		"""

		return (Psi, dPsidI)


class ParameterIdentificationTopologyOptimizationLoadcase(TopologyOptimizationLoadcase):

	def __init__(self, problem):

		raise Exception("implementation not up to date")

		self.problem = problem
		self.weight = Constant(self.getWeigth())
		self.optimality_criterion = AOptimalityCriterion()

		# init Dirichlet bc
		self.bc_u = self.getDirichletBC(problem.V_u)

		# check if bc are homogeneous
		self.homogeneous = True
		for bc in self.bc_u:
			if not np.all(np.array(bc.get_boundary_values().values()) == 0.0):
				self.homogeneous = False
				break
			
		if self.homogeneous:
			self.bc_u0 = self.bc_u
		else:
			self.bc_u0 = []
			for bc in self.bc_u:
				r = bc.value().value_rank()
				self.bc_u0.append(DirichletBC(bc.function_space(), Constant([0]*problem.mesh_gdim) if r > 0 else Constant(0), bc.user_sub_domain(), bc.method()))

		# init functions
		self.f = self.getBoundaryTraction(problem.V_f)
		self.g0 = self.getVolumeForce(problem.V_g)
		self.g1 = self.getVolumeGravity(problem.V_g)
		self.u = Function(problem.V_u)
		self.u_c = []
		self.adj_v = Function(problem.V_u)
		self.adj_vi = []
		for i, ci in enumerate(problem.material_model.params):
			self.u_c.append(Function(problem.V_u))
			self.adj_vi.append(Function(problem.V_u))

		n_p = len(problem.material_model.params)
		self.I = np.zeros((n_p, n_p), dtype=np.double)
		self.Psi = Constant(0.0)

		dx_ = problem.dx_
		ds_ = problem.ds_
		rho = problem.rho**problem.eta
		drho = problem.eta*(problem.rho**(problem.eta-1))

		def g(rho):
			return self.g0 + rho*self.g1

		#def s(rho):
		#	return (problem.rho**problem.eikonal_theta)

		# form for Fisher information matrix
		self.form_I = np.zeros((n_p, n_p), dtype=object)
		self.dPsi_dI = np.zeros((n_p, n_p), dtype=object)
		#w = ((problem.rho - problem.rho_min)/(1.0 - problem.rho_min))**(2*problem.eta)
		#w = Constant(1.0)
		w = (problem.rho)**(2*problem.eta)
		for i in range(n_p):
			for j in range(n_p):
				self.form_I[i,j] = w*dot(self.u_c[i], self.u_c[j])*dx_
				self.dPsi_dI[i,j] = Constant(0.0)

		# init forward problem
		v = TestFunction(problem.V_u)
		self.form_l = inner(self.f, v)*ds_ + inner(g(rho), v)*dx_
		self.mat_a = Matrix()
		self.vec_l = Vector()

		# u_ci
		self.form_l_u_c = []
		for i, ci in enumerate(problem.material_model.params):
			sigma_ci = diff(problem.material_model.sigma(eps(self.u), problem.mesh_gdim), ci)
			self.form_l_u_c.append(-rho*inner(sigma_ci, eps(v))*dx_)

		# adjoint v_i for u_ci
		du = TestFunction(problem.V_u)
		v = TrialFunction(problem.V_u)
		self.form_l_adj_vi = []
		for i, ci in enumerate(problem.material_model.params):
			# compute dPsidI : <dIdu_ci, du>
			F = None
			for j in range(n_p):
				dF = -2.0*self.dPsi_dI[i,j]*w*dot(self.u_c[j], du)*dx_
				F = dF if F is None else (F + dF)
			self.form_l_adj_vi.append(F)

		# adjoint for u
		self.form_l_adj_v = None
		for i, ci in enumerate(problem.material_model.params):
			sigma_ci = diff(problem.material_model.sigma(eps(du), problem.mesh_gdim), ci)
			dF = -rho*inner(sigma_ci, eps(self.adj_vi[i]))*dx_
			self.form_l_adj_v = dF if self.form_l_adj_v is None else (self.form_l_adj_v + dF)


		# objective and gradient
		# TODO: add Eikonal terms

		dr = TestFunction(problem.V_rho)
		self.form_F = self.weight*self.Psi/Constant(problem.mesh_V)*dx_(problem.mesh)
		self.form_dF = self.weight*drho*inner(problem.material_model.sigma(eps(self.u), problem.mesh_gdim), eps(self.adj_v))*dr*dx_ - self.weight*dot(diff(g(rho), problem.rho), self.adj_v)*dr*dx_
		for i, ci in enumerate(problem.material_model.params):
			sigma_ci = diff(problem.material_model.sigma(eps(self.u), problem.mesh_gdim), ci)
			self.form_dF += self.weight*drho*(inner(sigma_ci, eps(self.adj_vi[i])) + inner(problem.material_model.sigma(eps(self.u_c[i]), problem.mesh_gdim), eps(self.adj_vi[i])))*dr*dx_

		dw = diff(w, problem.rho)*dr
		#dw = Constant(0.0)*dr
		for i in range(n_p):
			for j in range(i, n_p):
				self.form_dF += (1.0 if (i == j) else 2.0)*self.weight*self.dPsi_dI[i,j]*dot(self.u_c[i], self.u_c[j])*dw*dx_(problem.mesh)


		return

		# init adjoint Eikonal linear form
		if isinstance(problem.material_model, TransverseIsotropicMaterialModel):

			self.eikonal_adj = Function(problem.V_eikonal)

			# add contribution to gradient
			self.form_dF += -self.weight*phi*problem.eikonal_theta*(problem.rho**(problem.eikonal_theta-1))*self.eikonal_adj*dx_

			u = TrialFunction(problem.V_eikonal)
			v = TestFunction(problem.V_eikonal)

			if 0:
				self.form_eikonal_adj_l = self.weight*rho/problem.eikonal_grad_norm*dot(problem.material_model.Bstar(eps(self.u), problem.eikonal_p, eps(self.uh), self.ascale, self.iscale, problem.mesh_gdim), grad(v))*dx_
			else:
				self.form_eikonal_adj_l = self.weight*rho/problem.eikonal_grad_norm*inner(problem.material_model.B(eps(self.u), problem.eikonal_p, grad(v), self.ascale, self.iscale, problem.mesh_gdim), eps(self.uh))*dx_


	def solveElasticity(self):

		problem = self.problem

		##########################################################
		# solve linear elasticity problem u
		
		assemble(self.form_l, self.vec_l)
		self.mat_a = problem.mat_a.copy()

		for bc in self.bc_u:
			bc.apply(self.mat_a)
			bc.apply(self.vec_l)

		solver_a = LUSolver(self.mat_a)
		solver_a.parameters["reuse_factorization"] = True
		solver_a.solve(self.u.vector(), self.vec_l)


		# solve sensitivities u_{\bar{c}_i} (depend on u)
		for i, u_c in enumerate(self.u_c):
			assemble(self.form_l_u_c[i], self.vec_l)
			for bc in self.bc_u0:
				bc.apply(self.vec_l)
			solver_a.solve(self.u_c[i].vector(), self.vec_l)
			
		# compute information matrix and optimality criterion (objective)
		for i in range(self.I.shape[0]):
			for j in range(i, self.I.shape[1]):
				self.I[i,j] = self.I[j,i] = assemble(self.form_I[i,j])
		
		Psi, dPsi_dI = self.problem.optimality_criterion.eval(self.I, derivative=True)
		self.Psi.assign(Psi)
		for i in range(self.I.shape[0]):
			for j in range(self.I.shape[1]):
				self.dPsi_dI[i,j].assign(dPsi_dI[i,j])
		
		# solve adjoint sensitivities \upsilon_i (depend on the u_{\bar{c}_i})
		for i, u_c in enumerate(self.u_c):
			assemble(self.form_l_adj_vi[i], self.vec_l)
			for bc in self.bc_u0:
				bc.apply(self.vec_l)
			solver_a.solve(self.adj_vi[i].vector(), self.vec_l)

		# solve adjoint elasticity \upsilon (depends on the \upsilon_i)
		assemble(self.form_l_adj_v, self.vec_l)
		for bc in self.bc_u0:
			bc.apply(self.vec_l)
		solver_a.solve(self.adj_v.vector(), self.vec_l)


class ParameterIdentificationTopologyOptimizationProblem(TopologyOptimizationProblem):
	def __init__(self):
		TopologyOptimizationProblem.__init__(self)
		app = QtGui.QApplication.instance()
		self.optimality_criterion = createInstance(app.parameters["optimality_criterion"], OptimalityCriterion)


class LineSearchAlgorithm(object):
	def __init__(self):
		pass
	def step(self, alpha_0, alpha_min, alpha_max):
		
		raise NotImplementedError()


class TopologyOptimizationAlgorithm(object):
	def __init__(self, problem):
		self.problem = problem
		self.params = []
	def init(self):
		pass
	# run the optimization algorithm
	def step(self):
		raise NotImplementedError()


class ProjectedGradientNewTopologyOptimizationAlgorithm(TopologyOptimizationAlgorithm):
	
	def __init__(self, problem):
		TopologyOptimizationAlgorithm.__init__(self, problem)
		app = QtGui.QApplication.instance()

	def init(self):
		self.rho_new = Function(self.problem.V_rho)
		self.rho_old = Function(self.problem.V_rho)
		self.dF_old = Function(self.problem.V_rho)
		self.rho_mu = Function(self.problem.V_rho)
		self.dF_mu = Function(self.problem.V_rho)
		self.tau_factor = 0.5
		self.flipper = True
		self.last_tau = self.tau_factor
		self.dF_scale = None
		self.mesh_id = 0

	def step(self):

		print ""

		tol = float(self.problem.tol)
		rho_bar = float(self.problem.rho_bar)
		rho_min = float(self.problem.rho_min)
		alpha_relax = float(self.problem.alpha_relax)
		rho = self.problem.rho.vector()
		dF = self.problem.dF.vector()
		rho_new = self.rho_new.vector()
		rho_old = self.rho_old.vector()
		dF_mu = self.dF_mu.vector()
		dF_old = self.dF_old.vector()
		rho_mu = self.rho_mu.vector()
		rho_mean = assemble(self.problem.rho*self.problem.dx_)/self.problem.mesh_V
		delta_rho = float(self.problem.delta_rho)
		alpha_0 = float(self.problem.alpha)

		rho_bar_next = rho_bar



			

		"""
		if abs(rho_mean - rho_bar) <= delta_rho:
			rho_bar_next = rho_bar
		elif rho_mean > rho_bar:
			rho_bar_next = rho_mean - delta_rho
		else:
			rho_bar_next = rho_mean + delta_rho

		# project the current density, otherwise we can not compare objectives
		self.problem.projectDensity(rho_bar=rho_bar_next)

		# normalize gradient
		dF_old_norm_inf = np.max(np.abs(dF.array()))
		dF_norm_2 = assemble(self.problem.dF**2*dx)**0.5
		dF_norm_inf = np.max(np.abs(dF.array()))

		if dF_old_norm_inf > 0:
			dF[:] /= dF_old_norm_inf
		else:
			dF[:] /= dF_norm_inf
		"""


		# compute current objective
		F_0 = self.problem.computeObjective()
		#rho[:] = rho_old[:]
		print "obj_old=", F_0


		alpha_min = 1e-9
		alpha_max = 1e9
		mu = 0.1
		tau = 0.5
		tol_red = 0.001

		alpha_0 = min(max(alpha_min, alpha_0), alpha_max)
		alpha = alpha_0


		rho_old[:] = rho[:]
		dF_old[:] = dF[:]
		converged = False


		def go_alpha(alpha):
			# perform step
			rho[:] = rho_old[:] - alpha_relax*alpha*dF_old[:]

			# determine Lagrange multiplier by bisection (project rho)
			self.problem.projectDensity(rho_bar=rho_bar_next)

		def eval_F_alpha(alpha):
			# perform step
			go_alpha(alpha)
			# compute objective
			self.problem.computeForward()
			F_alpha = self.problem.computeObjective()
			print "linesearch alpha=", alpha, " objective=", F_alpha
			return F_alpha
		

		def check_convergence():
			#conv_indicator = 2.0*(F_0 - F_alpha)/(F_0 + F_alpha)/tol_red
			conv_indicator = assemble(abs(self.rho_old - self.problem.rho)*dx)/self.problem.mesh_V
			converged = conv_indicator < tol
			print "convergence=", conv_indicator, converged
			return converged
		
		def step_change(alpha):
			go_alpha(alpha)
			return np.max(np.abs(rho.array() - rho_old.array()))


		if self.mesh_id != id(self.problem.mesh):
			# initialize step length if new mesh
			while True:
				sc = step_change(alpha)
				print "init step length alpha =", alpha, sc
				if sc > 0.5*(1 - rho_min):
					break
				alpha *= 2
			self.mesh_id = id(self.problem.mesh)

		# evaluate current step
		F_alpha = eval_F_alpha(alpha)
		alpha_new = alpha_min

		if F_alpha >= F_0:
			# reduce step
			F_min = F_0
			while alpha > alpha_min:
				converged = check_convergence()
				if converged:
					alpha_new = alpha
					break
				alpha *= tau
				F_alpha = eval_F_alpha(alpha)
				# check if step is better than last one
				if F_alpha < F_min:
					# check if we make not much progress
					if (F_min - F_alpha)/(F_0 - F_alpha) < tol_red:
						alpha_new = alpha
						break
					# save current minimum
					F_min = F_alpha

					# accept step (we do not want to reduce the step length uneccesarly)
					alpha_new = alpha
					break
				elif F_min < F_0 and F_alpha >= F_min:
					# step is actuall factor tau smaller than the optimal step, but we accept it
					# in order not to solve the forward problem again, etc.
					alpha_new = alpha/tau
					break
			# alpha = search_step(F_0, alpha, tau)
		else:
			# try larger step
			F_alpha_plus = eval_F_alpha(alpha/tau)
			if F_alpha_plus < F_alpha:
				alpha = alpha/tau
				alpha_new = alpha
				F_alpha = F_alpha_plus
				# search larger steps
				#alpha = search_step(F_alpha_plus, alpha, 1.0/tau)
			else:
				# try smaller step
				alpha *= tau
				F_alpha_minus = eval_F_alpha(alpha)
				if F_alpha_minus < F_alpha:
					alpha_new = alpha
					F_alpha = F_alpha_minus
					# search smaller steps
					#alpha = search_step(F_alpha_minus, alpha, tau)
				else:
					alpha_new = alpha/tau


		converged = check_convergence()


		# set the computed gradient/difference (for plots)
		dF[:] = (rho_old.array() - rho.array())/(alpha*alpha_relax)



		"""
		if alpha >= alpha_max:
			print "alpha >= alpha_max"
			converged = True
		
		if alpha <= alpha_min:
			print "alpha <= alpha_min"
			converged = True
		"""


		# save step
		self.problem.alpha.assign(alpha_new)

		print "rho_bar=", assemble(self.problem.rho*dx)/self.problem.mesh_V
		#print "max dF=", np.max(dF.array())
		#print "dF_norm_2=", dF_norm_2, "dF_norm_inf=", dF_norm_inf
		print "alpha=", alpha, "alpha_new=", alpha_new

		return converged


	def _step_new(self):

		rho_bar = float(self.problem.rho_bar)
		rho_min = float(self.problem.rho_min)
		rho = self.problem.rho.vector()
		dF = self.problem.dF.vector()
		rho_new = self.rho_new.vector()
		rho_old = self.rho_old.vector()
		dF_mu = self.dF_mu.vector()
		dF_old = self.dF_old.vector()
		rho_mu = self.rho_mu.vector()
		rho_mean = assemble(self.problem.rho*self.problem.dx_)/self.problem.mesh_V
		delta_rho = float(self.problem.delta_rho)

		if abs(rho_mean - rho_bar) <= delta_rho:
			rho_bar_next = rho_bar
		elif rho_mean > rho_bar:
			rho_bar_next = rho_mean - delta_rho
		else:
			rho_bar_next = rho_mean + delta_rho

		# project the current density, otherwise we can not compare objectives
		self.problem.projectDensity(rho_bar=rho_bar_next)

		# compute the grandient
		self.problem.computeGradient()

		# normalize gradient
		dF_norm2 = assemble(self.problem.dF**2*dx)
		dF[:] /= dF_norm2**0.5
		dF_norm2 = 1.0

		# compute current objective
		F_0 = self.problem.computeObjective()
		#rho[:] = rho_old[:]
		print "obj_old=", F_0


		alpha_0 = float(self.problem.alpha)
		alpha_min = 1e-9
		alpha_max = 1e9
		mu = 0.1
		tau = 0.5

		alpha_0 = min(max(alpha_min, alpha_0), alpha_max)
		alpha = alpha_0


		rho_old[:] = rho[:]
		dF_old[:] = dF[:]
		converged = False

		print ""

		def eval_F_alpha(alpha):
			# perform step
			rho[:] = rho_old[:] - alpha*dF_old[:]

			# determine Lagrange multiplier by bisection (project rho)
			self.problem.projectDensity(rho_bar=rho_bar_next)

			# compute objective
			self.problem.computeForward()
			F_alpha = self.problem.computeObjective()
			print "linesearch alpha=", alpha, " objective=", F_alpha, " ls_goal=", F_0 - mu*alpha*dF_norm2
			return F_alpha
		
		def test_Armijo(alph, F_alpha):
			# Armijo test
			if F_alpha <= F_0 - mu*alpha*dF_norm2:
				# accept step length
				return True
			return False

		while True:
			# Armijo test
			F_alpha = eval_F_alpha(alpha)
			if test_Armijo(alpha, F_alpha):
				# accept step length
				break

			# reduce step length
			alpha *= tau		

			# test if below minimal step length
			if (alpha <= alpha_min):
				# accept minimal step length, even if Armijo test fails
				alpha = alpha_min
				break

		
		# test if initial step length was accepted
		if alpha == alpha_0:
			# remember current (best) objective
			F_alpha_0 = F_alpha
			# try to increase step length unitl Armijo fails
			alpha = min(alpha/tau, alpha_max)
			while True:
				# Armijo test
				F_alpha = eval_F_alpha(alpha)
				if not test_Armijo(alpha, F_alpha) or (F_alpha >= F_alpha_0):
					# accept previous step
					alpha = max(alpha_min, tau*alpha)
					# evaluate step again
					eval_F_alpha(alpha)
					break

				# increase step length
				alpha /= tau
				
				# test if above maximal step length
				if (alpha >= alpha_max):
					# accept maximal step length, even if Armijo test fails
					alpha = alpha_max
					break

		
		# set the computed gradient (for plots)
		dF[:] = (rho_old.array() - rho.array())/alpha


		"""
		if alpha >= alpha_max:
			print "alpha >= alpha_max"
			converged = True
		
		if alpha <= alpha_min:
			print "alpha <= alpha_min"
			converged = True
		"""


		# save step
		self.problem.alpha.assign(alpha)

		print "rho_bar=", assemble(self.problem.rho*dx)/self.problem.mesh_V
		print "max dF=", np.max(dF.array())
		print "alpha=", alpha

		return converged


	def _step_old(self):

		mu0 = float(self.problem.alpha)
		alpha = float(self.problem.alpha_relax)
		rho_bar = float(self.problem.rho_bar)
		rho_min = float(self.problem.rho_min)
		rho = self.problem.rho.vector()
		dF = self.problem.dF.vector()
		rho_new = self.rho_new.vector()
		rho_old = self.rho_old.vector()
		dF_mu = self.dF_mu.vector()
		dF_old = self.dF_old.vector()
		rho_mu = self.rho_mu.vector()
		rho_mean = assemble(self.problem.rho*self.problem.dx_)/self.problem.mesh_V
		delta_rho = float(self.problem.delta_rho)

		if abs(rho_mean - rho_bar) <= delta_rho:
			rho_bar_next = rho_bar
		elif rho_mean > rho_bar:
			rho_bar_next = rho_mean - delta_rho
		else:
			rho_bar_next = rho_mean + delta_rho

		# project the current density, otherwise we can not compare objectives
		self.problem.projectDensity(rho_bar=rho_bar_next)

		# compute the grandient
		self.problem.computeGradient()

		obj_old = self.problem.computeObjective()
		#rho[:] = rho_old[:]
		print "obj_old=", obj_old

		# normalize gradient
		"""
		if self.dF_scale is None:
			self.dF_scale = 1.0/np.max(np.abs(dF.array()))
		dF[:] *= self.dF_scale
		"""

		dF_norm2 = assemble(self.problem.dF**2*dx)
		dF[:] /= dF_norm2**0.5

		rho_old[:] = rho[:]
		dF_old[:] = dF[:]

		mu_list = [mu0, self.last_tau*mu0, mu0/self.last_tau]
		obj_min = float('inf')
		obj_cur = obj_min
		mu_min = None
		mu_scale = float(self.problem.alpha_scale)
		tol_red = 0.05
		converged = False

		while True:
			
			# perfor step length search
			for mu in mu_list:

				# perform step
				rho[:] = rho_old[:] - (mu_scale*mu)*dF_old[:]

				# determine Lagrange multiplier by bisection (project rho)
				self.problem.projectDensity(rho_bar=rho_bar_next)

				# normalize
				#rho_new[:] *= 1.0/np.max(np.abs(rho_new.array()))

				# save projected gradient
				dF[:] = rho_old[:] - rho[:]

				# perform step
				rho[:] = rho_old[:] - alpha*dF[:]

				# compute objective
				self.problem.computeForward()
				obj = self.problem.computeObjective()
				print "mu=", mu, "objective=", obj

				if obj < obj_min:
					obj_min = obj
					mu_min = mu
					rho_mu[:] = rho[:]
					dF_mu[:] = dF[:]
					if mu != mu0:
						# found a new nu with smaller objective, no need to check the remaining nu
						break

			if obj_min >= obj_old:
				# no descent direction found, decrease mu0
				mu0 = min(mu_list)
				if mu0 < self.problem.eps:
					# converged
					print "solution converged!"
					converged = True
					break
				mu_list = [mu0*self.tau_factor]
			else:
				# have descent direction
				if mu_min != mu0:
					# try to reduce objective even more (until it becomes inefficient)
					if obj_min < obj_cur:
						relative_reduction = (obj_cur - obj_min)/(obj_old - obj_min)
						if relative_reduction < tol_red:
							break
						tau = mu_min/mu0
						mu0 = mu_min
						mu_list = [mu_min*tau]
						obj_cur = obj_min
					else:
						break
				else:
					break

		# use best step
		rho[:] = rho_mu[:]
		dF[:] = dF_mu[:]
		if mu_min != mu0:
			self.last_tau = self.tau_factor if mu_min < mu0 else (1.0/self.tau_factor)
		self.problem.computeForward()

		# save step
		self.problem.alpha.assign(mu_min)

		#mu = 0.2*mu/np.max(dF.array()) + 0.8*mu
		#self.problem.alpha.assign(mu)

		#dF_mu = np.max(dF.array())/mu_min
		dF_alpha = np.max(dF.array())/alpha

		print "rho_bar=", assemble(self.problem.rho*dx)/self.problem.mesh_V
		print "max dF=", np.max(dF.array())
		print "mu=", mu_min
		print "tau=", self.last_tau

		if dF_alpha < rho_min:
			print "converged dF_alpha < rho_min"

		#print "alpha=", mu

		return converged


class BoostedTopologyOptimizationAlgorithm(TopologyOptimizationAlgorithm):
	
	def __init__(self, problem):
		TopologyOptimizationAlgorithm.__init__(self, problem)
		app = QtGui.QApplication.instance()

	def init(self):
		self.rho_old = Function(self.problem.V_rho)

	def step(self):

		eps = 1e-2
		app = QtGui.QApplication.instance()
		alpha_max = 1e6 # float('inf') # float(self.problem.alpha_conv)
		alpha_relax = float(self.problem.alpha_relax)
		tol = float(self.problem.tol)
		alpha = float(self.problem.alpha)
		rho_min = float(self.problem.rho_min)
		rho_old = self.rho_old
		rho = self.problem.rho
		rho_old.assign(rho)

		def perform_step(alpha):
			rho.assign(rho_old)
			rho.vector().axpy(-alpha, self.problem.dF.vector())
			self.problem.projectDensity()

		def func(alpha):
			perform_step(alpha)
			rho.vector().axpy(-1.0, rho_old.vector())
			ret = np.max(np.abs(rho.vector().array())) - alpha_relax*(0.5-2*eps)*(1 - rho_min)
			#print "f", alpha, ret
			return ret

		def fp_func(alpha):
			perform_step(alpha)
			rho.vector().axpy(-1.0, rho_old.vector())
			ret = alpha*alpha_relax*0.499*(1 - rho_min)/np.max(np.abs(rho.vector().array()))
			return ret

		#sys.stdout.write("step length calculation...")

		converged = False

		if 0:
			# fixed point method
			alpha_old = alpha + 2*eps
			alpha_old = eps
			while abs(alpha - alpha_old) > eps:
				alpha_old = alpha
				alpha = fp_func(alpha_old)
				print "fp_alpha", alpha
		else:
			# bisection
			alpha_min = 0.0

			# increase bound alpha until func(alpha) > 0
			while (alpha < alpha_max) and (func(alpha) <= 0.0):
				#print "reduce", alpha, func(alpha)
				alpha_min = alpha
				alpha *= 2.0
				#print "increasing alpha=", alpha
			
			if alpha < alpha_max:
				alpha_old = alpha
				alpha = bisect(func, alpha_min, alpha, rtol=eps)
			else:
				alpha = alpha_max
				converged = True

			#print "bs_alpha", alpha, alpha_old, alpha_min

		#print "DONE"
		
		perform_step(alpha)
		self.problem.alpha.assign(alpha)

		if not converged:
			conv_measure = assemble(abs(rho - rho_old)*dx)/self.problem.mesh_V
			print "conv_measure=", conv_measure
			if conv_measure < tol:
				converged = True

		return converged


class ArmijoTopologyOptimizationAlgorithm(TopologyOptimizationAlgorithm):
	
	def __init__(self, problem):
		TopologyOptimizationAlgorithm.__init__(self, problem)
		app = QtGui.QApplication.instance()
		self.boosted = BoostedTopologyOptimizationAlgorithm(self.problem)

	def init(self):
		self.rho_old = Function(self.problem.V_rho)
		self.dF_old_norm_2 = None
		self.boosted.init()

	def step(self):

		eps = 1e-2
		app = QtGui.QApplication.instance()
		alpha_min = 1e-9
		alpha_max = 1e6 # float('inf') # float(self.problem.alpha_conv)
		alpha_relax = float(self.problem.alpha_relax)
		alpha = float(self.problem.alpha)
		rho_min = float(self.problem.rho_min)
		rho_old = self.rho_old.vector()
		rho = self.problem.rho.vector()
		dF = self.problem.dF.vector()
		rho_old[:] = rho[:]

		converged = False
		mu = 0.001
		tau = 0.5

		dF_norm_2 = assemble(self.problem.dF**2*dx)
		F_0 = self.problem.computeObjective()

		# initial gues for step length
		if self.dF_old_norm_2 is None:
			self.boosted.step()
			alpha = float(self.problem.alpha)
		else:
			# alpha*self.dF_old_norm_2 = alpha_new*dF_norm_2
			alpha = alpha*self.dF_old_norm_2/dF_norm_2

		self.dF_old_norm_2 = dF_norm_2

		sys.stdout.write("step length calculation...")

		def eval_F_alpha(alpha):

			# perform step
			rho[:] = rho_old[:] - alpha*dF[:]

			# determine Lagrange multiplier by bisection (project rho)
			self.problem.projectDensity()

			# compute objective
			self.problem.computeForward()
			F_alpha = self.problem.computeObjective()
			print "linesearch alpha=", alpha, " objective=", F_alpha, " ls_goal=", F_0 - mu*alpha*dF_norm_2
			return F_alpha
		
		def test_Armijo(alph, F_alpha):
			# Armijo test
			if F_alpha <= F_0 - mu*alpha*dF_norm_2:
				# accept step length
				return True
			return False

		while True:
			# Armijo test
			F_alpha = eval_F_alpha(alpha)
			if test_Armijo(alpha, F_alpha):
				# accept step length
				break

			# reduce step length
			alpha *= tau		

			# test if below minimal step length
			if (alpha <= alpha_min):
				# accept minimal step length, even if Armijo test fails
				alpha = alpha_min
				converged = True
				break

		print "DONE"
		
		self.problem.alpha.assign(alpha)

		return converged



class ProjectedGradientTopologyOptimizationAlgorithm(TopologyOptimizationAlgorithm):
	
	def __init__(self, problem):
		TopologyOptimizationAlgorithm.__init__(self, problem)
		app = QtGui.QApplication.instance()

	def init(self):
		self.obj_old = None
		self.step_num = 0

	def step(self):

		#self.problem.computeGradient()
		obj = self.problem.computeObjective()
		#print "objective=", obj

		#plot(self.problem.rho, interactive=True)
		#plot(self.problem.u, interactive=True)
		
		# perform step
		#dF_norm2 = assemble(self.problem.dF**2*dx)
		alpha = float(self.problem.alpha) # /dF_norm2
		rho = self.problem.rho.vector()
		rho.axpy(-alpha, self.problem.dF.vector())
		self.problem.projectDensity()

		converged = not self.obj_old is None and obj > self.obj_old and self.step_num > 10
		self.obj_old = obj
		self.step_num += 1

		return converged


class GuideWeightTopologyOptimizationAlgorithm(TopologyOptimizationAlgorithm):
	
	def __init__(self, problem):
		TopologyOptimizationAlgorithm.__init__(self, problem)
		app = QtGui.QApplication.instance()

	def init(self):
		self.rho_new = Function(self.problem.V_rho)

	def step(self):

		self.problem.computeGradient()

		obj = self.problem.computeObjective()
		print "objective=", obj

		def proj(expr, a, b):
			return conditional(gt(expr, b), b, conditional(lt(expr, a), a, expr))

		int_dF = Constant(assemble(self.problem.form_int_dF)/self.problem.mesh_V)

		print "int dF=", float(int_dF)

		rho = self.problem.rho


		print "rho_bar=", assemble(rho*self.problem.dx_)/self.problem.mesh_V, "rho_min=", np.min(rho.vector().array()), "rho_max=", np.max(rho.vector().array())

		project(conditional(ge(rho, 1.0-DOLFIN_EPS), 1.0, conditional(le(rho, self.problem.rho_min+DOLFIN_EPS), self.problem.rho_min,
			self.problem.alpha*(self.problem.rho_bar*self.problem.dF/(int_dF)) + (1.0-self.problem.alpha)*rho)), function=self.rho_new)

		rho.vector()[:] = np.minimum(np.maximum(self.rho_new.vector().array(), float(self.problem.rho_min)), 1.0)

		print "rho_bar=", assemble(rho*self.problem.dx_)/self.problem.mesh_V, "rho_min=", np.min(rho.vector().array()), "rho_max=", np.max(rho.vector().array())

		#rho.vector()[:] = rho.vector().array()*self.problem.dF_weights.vector().array()


# https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/
class NesterovsMethodTopologyOptimizationAlgorithm(TopologyOptimizationAlgorithm):
	
	def __init__(self, problem):
		TopologyOptimizationAlgorithm.__init__(self, problem)
		app = QtGui.QApplication.instance()

	def init(self):
		self.x = self.problem.rho
		self.y = Function(self.problem.V_rho)
		self.y_old = Function(self.problem.V_rho)
		self.lmbda_old = 0.0
		self.beta = 1.0	# = Lipschitz Constant
		self.y.assign(self.x)
		self.y_old.assign(self.x)

	def step(self):

		x = self.x.vector()
		y = self.y.vector()
		y_old = self.y_old.vector()

		self.problem.computeGradient()
		obj = self.problem.computeObjective()
		print "objective=", obj

		#plot(self.problem.rho, interactive=True)
		#plot(self.problem.u, interactive=True)
		
		# perform step
		lmbda = 0.5*(1.0 + (1.0 + 4.0*self.lmbda_old**2)**0.5)
		lmbda_next = 0.5*(1.0 + (1.0 + 4.0*lmbda**2)**0.5)
		gamma = (1.0 - lmbda)/lmbda_next
		y[:] = x.array() - (1.0/self.beta)*self.problem.dF.vector().array()
		x[:] = (1.0 - gamma)*y.array() + gamma*y_old.array()

		self.y_old.assign(self.y)
		self.lmbda_old = lmbda


class DiscreteSetTopologyOptimizationAlgorithm(TopologyOptimizationAlgorithm):
	
	def __init__(self, problem):
		TopologyOptimizationAlgorithm.__init__(self, problem)
		app = QtGui.QApplication.instance()

		if app.parameters["rho_space"] != "DG0":
			raise RuntimeError("the discrete set algorithm requires DG0 elements")
		

	def init(self):
		if not hasattr(self.problem, "computeGradientOrg"):
			self.problem.computeGradientOrg = self.problem.computeGradient
		self.problem.computeGradient = lambda rescale=False: self.problem.computeGradientOrg(False)

		# reimplement mesh refinement
		# self.problem.refine = 

	def step(self):

		self.problem.computeGradient()

		dF = self.problem.dF.vector()
		rho = self.problem.rho.vector()
		dFa = dF.array()
		rhoa = rho.array()

		# rescale (inverse mass matrix)
		Ainv = self.problem.dF_weights.vector().array()
		A = Ainv**(-1)
		dFa *= Ainv

		S = np.arange(len(dFa))
		Sa = S
		#S0 = np.intersect1d(Sa, np.where(np.logical_and(rhoa <  1.0, dFa < 0))[0], True)
		S0 = np.intersect1d(Sa, np.where(rhoa <  1.0)[0], True)
		I = np.argsort(dFa[S0]); S0 = S0[I]
		#S1 = np.intersect1d(Sa, np.where(np.logical_and(rhoa == 1.0, dFa > 0))[0], True)
		S1 = np.intersect1d(Sa, np.where(rhoa == 1.0)[0], True)
		I = np.argsort(-dFa[S1]); S1 = S1[I]
		S01 = np.union1d(S0, S1)
		Sc = np.setdiff1d(S, S01)


		dFa[Sc] = 0
		dF[:] = dFa
		

		V_bar = float(self.problem.rho_bar)*self.problem.mesh_V
		r = np.sum(A[S1]) - V_bar
		n0 = 0
		n1 = 0
		dA = 0
		dAmax = self.problem.alpha*V_bar

		while (n0 < len(S0) and n1 < len(S1)):
			
			ra = abs(r)
			r0 = abs(r + A[S0[n0]])
			r1 = abs(r - A[S1[n1]])

			print ra, r0, r1

			if r0 < r1:
				r += A[S0[n0]]
				dA += A[S0[n0]]
				rho[S0[n0]] = 1
				n0 += 1
			else:
				r -= A[S1[n1]]
				dA += A[S1[n1]]
				rho[S1[n1]] = float(self.problem.rho_min)
				n1 += 1

			if dA > dAmax:
				break


		obj = self.problem.computeObjective()
		print "objective=", obj


# python run.py --problem=TGradCompare --material_model=Isotropic
class TGradCompareComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def getMesh(self, R):
		H = 1.0 
		W = 1.0 
		T = 0.3*H
		res = R
		domain = mshr.Rectangle(Point(0.0, 0.0), Point(W, H)) 
		#domain1 = mshr.Rectangle(Point(0.0, 0.5*(H-T)), Point(W, 0.5*(H+T)))
		domain1 = mshr.Circle(Point(0.5*W, 0.5*H), T, int(4*res))
		domain0 = domain - domain1
		domain = domain0 + domain1
		return domain

	def getInitialRho(self, V):
		return None
		rho = interpolate(Expression("(x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) <= 0.3*0.3 ? 1.0 : rho_min", rho_min=float(self.rho_min), degree=1), V)
		return rho

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && x[0] < DOLFIN_EPS"))]
		def getVolumeForce(self, V):
			return Constant([0, 0])
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getBoundaryTraction(self, V):
			return Constant([0, -1])

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]


class SelfSupportingBeamComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def getMesh(self, R):
		if 0:
			return RectangleMesh(Point(0, 0), Point(2, 1), 2*R, R)
		domain = mshr.Rectangle(Point(0, 0), Point(2, 1))
		return domain

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && x[0] < DOLFIN_EPS && x[1] < 0.5")),
			DirichletBC(V, Constant([0, 1]), CompiledSubDomain("on_boundary && x[0] >= 2-DOLFIN_EPS"))]
		def getVolumeForce(self, V):
			return Constant([0, 1])
		def getVolumeGravity(self, V):
			return Constant([0, -1])
			return Constant([0, 0])
			return Expression(["0", "((x[0]-1.0)*(x[0]-1.0) + (x[1]-0.5)*(x[1]-0.5) < 0.1) ? -1 : 0"], degree=1)
			return Expression(["1", "0"], degree=1)
		def getBoundaryTraction(self, V):
			return Constant([0, 0])
			return Expression(["(x[0] >= 2-DOLFIN_EPS) ? -1 : 0", "0"], degree=1)
			return Expression(["0", "(x[0] >= 2-DOLFIN_EPS) ? -1 : 0"], degree=1)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]


class Box3dComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
	def getMesh(self, R):
		R = int(R/4+1)
		return BoxMesh(Point(0, 0, 0), Point(2, 1, 1), 2*R, R, R)

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [DirichletBC(V, Constant([0, 0, 0]), CompiledSubDomain("on_boundary && x[0] < DOLFIN_EPS && x[1] < 0.5")),
			DirichletBC(V, Constant([0, 1, 0]), CompiledSubDomain("on_boundary && x[0] >= 2-DOLFIN_EPS"))]
		def getVolumeForce(self, V):
			return Constant([0, -1, 0])
		def getVolumeGravity(self, V):
			return Constant([0, 0, 0])
			return Constant([0, 0, 0])
			return Expression(["0", "((x[0]-1.0)*(x[0]-1.0) + (x[1]-0.5)*(x[1]-0.5) < 0.1) ? -1 : 0", "0"], degree=1)
			return Expression(["1", "0", "0"], degree=1)
		def getBoundaryTraction(self, V):
			return Constant([0, 0, 0])
			return Expression(["(x[0] >= 2-DOLFIN_EPS) ? -1 : 0", "0", "0"], degree=1)
			return Expression(["0", "(x[0] >= 2-DOLFIN_EPS) ? -1 : 0", "0"], degree=1)
	
	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]

class GravityComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 1
		self.geometry_L = 1
		self.geometry_H = 1
	def getMesh(self, R):
		#domain = mshr.Surface3D("../meshes/plate_w_inset.stl")
		domain = mshr.Rectangle(Point(0, 0), Point(self.geometry_L, self.geometry_H))
		return domain
	def getInitialRho(self, V):
		rho = interpolate(Expression("x[0] < rho_bar ? 1.0 : rho_min", rho_bar=float(self.rho_bar), rho_min=float(self.rho_min), degree=1), V)
		return rho
	def getInjectionDomain(self):
		return CompiledSubDomain("x[0] < DOLFIN_EPS")

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, -1])
		def getBoundaryTraction(self, V):
			return Constant([0, 0])
			return Expression(["(x[0] >= L-DOLFIN_EPS) ? 1 : 0", "0"], L=self.problem.geometry_L, degree=1)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]


# --material_model=Simple --rho_bar=0.6
class MBBBeamComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 90/20
		self.geometry_L = 90
		self.geometry_H = 30
	def getMesh(self, R):
		domain = mshr.Rectangle(Point(0, 0), Point(self.geometry_L, self.geometry_H))
		return domain
		return RectangleMesh(Point(0, 0), Point(self.geometry_L, self.geometry_H), int(R*self.geometry_L+0.5), int(R*self.geometry_H+0.5))
	def getInjectionDomain(self):
		return CompiledSubDomain("x[0] < DOLFIN_EPS")

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V.sub(0), Constant(0), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			#DirichletBC(V.sub(1), Constant(0), CompiledSubDomain("on_boundary && (x[0] >= L-DOLFIN_EPS) && (x[1] <= 2+DOLFIN_EPS)", L=self.problem.geometry_L)),
			DirichletBC(V.sub(1), Constant(0), CompiledSubDomain("on_boundary && (x[0] >= L-2-DOLFIN_EPS) && (x[1] <= DOLFIN_EPS)", L=self.problem.geometry_L)),
			]
		def getBoundaryTraction(self, V):
			return Expression(["0", "(x[0] <= 2+DOLFIN_EPS) && (x[1] >= H-DOLFIN_EPS) ? -1.0 : 0"],
				L=self.problem.geometry_L, H=self.problem.geometry_H, degree=1)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]


class TBeamComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 1
		self.geometry_L = 2
		self.geometry_H = 1
	def getMesh(self, R):
		if 0:
			return RectangleMesh(Point(0, 0), Point(1, 1), int(R), int(R))
		#domain = mshr.Surface3D("../meshes/plate_w_inset.stl")
		domain = mshr.Rectangle(Point(0, 0), Point(self.geometry_L, self.geometry_H))
		return domain
	def getInitialRho(self, V):
		D = bisect(lambda D: float(self.rho_bar) - (D*self.geometry_L + D*self.geometry_H - D*D)/(self.geometry_L*self.geometry_H) -
			(self.geometry_L*self.geometry_H - (D*self.geometry_L + D*self.geometry_H - D*D))/(self.geometry_L*self.geometry_H)*float(self.rho_min),
			0.0, max(self.geometry_L, self.geometry_H))
		rho = interpolate(Expression("(fabs(x[1] - 0.5*H) < 0.5*D || (L - x[0]) < D) ? 1.0 : rho_min",
			L=self.geometry_L, H=self.geometry_H, D=D, rho_min=float(self.rho_min), degree=1), V)
		return rho
		return None
	def getInjectionDomain(self):
		return CompiledSubDomain("x[0] < DOLFIN_EPS")
		return CompiledSubDomain("pow(fabs(x[0]-0.5), 2.0) + pow(fabs(x[1]-0.5), 2.0) < 0.01")
		return CompiledSubDomain("(x[0] < DOLFIN_EPS || x[0] >= 1-DOLFIN_EPS)")

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			#DirichletBC(V, Constant([0, 1]), CompiledSubDomain("on_boundary && (x[0] >= L-DOLFIN_EPS)", L=self.problem.geometry_L)),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, 0])
		def getBoundaryTraction(self, V):
			return Expression(["0", "(x[0] >= L-DOLFIN_EPS && fabs(x[1] - 0.5*H) <= 0.05*H) ? -1 : 0"], L=self.problem.geometry_L, H=self.problem.geometry_H, degree=1)
			return Expression(["(x[0] >= L-DOLFIN_EPS) ? 1 : 0", "0"], L=self.problem.geometry_L, degree=1)

	class Loadcase2(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			#DirichletBC(V, Constant([0, 1]), CompiledSubDomain("on_boundary && (x[0] >= L-DOLFIN_EPS)", L=self.problem.geometry_L)),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, 0])
		def getBoundaryTraction(self, V):
			return Expression(["(x[0] >= L-DOLFIN_EPS && fabs(x[1] - 0.5*H) <= 0.05*H) ? -1 : 0", "0"], L=self.problem.geometry_L, H=self.problem.geometry_H, degree=1)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
			self.Loadcase2(self),
		]


class GAMMComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 1
		self.image = scipy.misc.imread("GAMM_gray.png")
		self.load = np.sum(self.image, axis=0)/255.0
		self.load *= 1.0/np.max(self.load)
		self.aspect = self.image.shape[1]/float(self.image.shape[0])
	def getMesh(self, R):
		if 1:
			mesh = RectangleMesh(Point(0, 0), Point(self.aspect, 1), int(R*self.aspect), int(R))
			mesh = refine(mesh)
			return mesh
		domain = mshr.Rectangle(Point(0, 0), Point(self.aspect, 1))
		return domain
	def getInitialRho(self, V):
		return None
	def getInjectionDomain(self):
		return CompiledSubDomain("(x[1] < DOLFIN_EPS)")

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		class MyExpression0(Expression):
			def __init__(self, problem, degree):
				self.problem = problem
			def eval(self, value, x):
				i = min(self.problem.image.shape[0]-1, int(x[1]*self.problem.image.shape[0]))
				j = min(self.problem.image.shape[1]-1, int(x[0]/self.problem.aspect*self.problem.image.shape[1]))
				value[0] = 0.0
				value[1] = self.problem.image[-i,j]/255 - 1
			def value_shape(self):
				return (2,)
		class MyExpression1(Expression):
			def __init__(self, problem, degree):
				self.problem = problem
			def eval(self, value, x):
				j = min(self.problem.image.shape[1]-1, int(x[0]/self.problem.aspect*self.problem.image.shape[1]))
				value[0] = 0.0;
				value[1] = -self.problem.load[j]
			def value_shape(self):
				return (2,)
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[1] < DOLFIN_EPS)")),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
			return self.MyExpression0(problem=self.problem, degree=1)
		def getVolumeForce(self, V):
			return Constant([0, 0])
		def getBoundaryTraction(self, V):
			return self.MyExpression1(problem=self.problem, degree=1)
			return Constant([0, 0])

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]



class SeatComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 4
		self.geometry_W0 = 0.6
		self.geometry_W1 = 0.5
		self.geometry_H0 = 1.0
		self.geometry_H1 = 0.5
	def getMesh(self, R):
		domain = mshr.Rectangle(Point(0, 0), Point(self.geometry_W0, self.geometry_H0)) - \
			mshr.Rectangle(Point(self.geometry_W0-self.geometry_W1, self.geometry_H0-self.geometry_H1), Point(self.geometry_W0, self.geometry_H0))
		return domain
	def getInitialRho(self, V):
		return None
	def getInjectionDomain(self):
		return CompiledSubDomain("on_boundary && x[1] >= H0 - DOLFIN_EPS", H0=self.geometry_H0)

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[1] < DOLFIN_EPS)")),
			#DirichletBC(V.sub(1), Constant(0), CompiledSubDomain("on_boundary && (x[1] < DOLFIN_EPS)")),
			#DirichletBC(V.sub(0), Constant(0), CompiledSubDomain("on_boundary && (x[0] > W0-W1 + DOLFIN_EPS) && (x[1] >= H0-H1 - DOLFIN_EPS)", W0=self.problem.geometry_W0, W1=self.problem.geometry_W1, H0=self.problem.geometry_H0, H1=self.problem.geometry_H1)),
			]
		def getBoundaryTraction(self, V):
			return Expression(["(x[0] >= W0-W1 - DOLFIN_EPS) && (x[1] > H0-H1 + DOLFIN_EPS) ? -0.2*(x[1]-(H0-H1))/H1 : 0",
				"(x[0] > W0-W1 + DOLFIN_EPS) && (x[1] >= H0-H1 - DOLFIN_EPS) ? -1 : 0"], degree=1,
				W0=self.problem.geometry_W0, W1=self.problem.geometry_W1,
				H0=self.problem.geometry_H0, H1=self.problem.geometry_H1)

	class Loadcase2(Loadcase1):
		def getBoundaryTraction(self, V):
			return Expression(["(x[0] > W0-W1 + DOLFIN_EPS) && (x[1] >= H0-H1 - DOLFIN_EPS) ? 0.4 : 0", "0"], degree=1,
				W0=self.problem.geometry_W0, W1=self.problem.geometry_W1,
				H0=self.problem.geometry_H0, H1=self.problem.geometry_H1)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
			self.Loadcase2(self),
		]



# python run.py --problem=Tip --rho_init=constant --material_model=Simple --initial_refinements=1 --rho0=0.5 --rho_bar=0.5 --penalty=0.5 --ip_grad_scale=1 --alpha_relax=0.5
class TipComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 4
		self.geometry_L = 4.0
	def getMesh(self, R):
		if 0:
			mesh = RectangleMesh(Point(0, 0), Point(self.geometry_L + 1e-10, 1), int(R*self.geometry_L), int(R))
			mesh = refine(mesh)
			return mesh
		domain = mshr.Rectangle(Point(0, 0), Point(self.geometry_L, 1))
		return domain
	def getInitialRho(self, V):
		return None
	def getInjectionDomain(self):
		return CompiledSubDomain("on_boundary && (x[0] >= 0.5*L) && (fabs(x[1]-0.5) <= 0.05)", L=self.geometry_L)
		return CompiledSubDomain("(x[0] <= DOLFIN_EPS && x[1] <= 0.05)")
		return CompiledSubDomain("(x[1] >= 1-DOLFIN_EPS)")

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			#DirichletBC(V.sub(1), Constant(0), CompiledSubDomain("on_boundary && (x[1] >= 1-DOLFIN_EPS || x[1] <= DOLFIN_EPS)")),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, 0])
		#def getPointSources(self, V):
		#	return [PointSource(V.sub(0), Point(self.problem.geometry_L-DOLFIN_EPS, 0.5), 1.0)]
		def getBoundaryTraction(self, V):
		#	return Constant([0, 0])
			return Expression(["(x[0] >= L-DOLFIN_EPS) && (fabs(x[1]-0.5) <= 0.05) ? 1 : 0", "0"], degree=1, L=self.problem.geometry_L)

	class Loadcase2(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, 0])
		#def getPointSources(self, V):
		#	return [PointSource(V.sub(1), Point(self.problem.geometry_L-DOLFIN_EPS, 0.5), -1.0)]
		def getBoundaryTraction(self, V):
		#	return Constant([0, 0])
			return Expression(["0", "(x[0] >= L-DOLFIN_EPS) && (fabs(x[1]-0.5) <= 0.05) ? -1 : 0"], degree=1, L=self.problem.geometry_L)
		def getWeigth(self):
			return 1.0

	class Loadcase3(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, 0])
		def getBoundaryTraction(self, V):
			return Expression(["(x[0] >= L-DOLFIN_EPS) && (fabs(x[1]-0.5) <= 0.05) ? 1 : 0",
				"(x[0] >= L-DOLFIN_EPS) && (fabs(x[1]-0.5) <= 0.05) ? -1 : 0"], degree=1, L=self.problem.geometry_L)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
			self.Loadcase2(self),
			#self.Loadcase3(self),
		]


# python run.py --problem=Tip --rho_init=constant --material_model=Simple --initial_refinements=1 --rho0=0.5 --rho_bar=0.5 --penalty=0.5 --ip_grad_scale=1 --alpha_relax=0.5
class Tip3DComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.geometry_L = 4.0
	def getMesh(self, R):
		mesh = BoxMesh(Point(0, 0, 0), Point(self.geometry_L + 1e-8, 1, 1), int(R*self.geometry_L), int(R), int(R))
		mesh = refine(mesh)
		return mesh
	def getInitialRho(self, V):
		return None
	def getInjectionDomain(self):
		return CompiledSubDomain("(x[0] >= L-DOLFIN_EPS) && (fabs(x[1]-0.5) <= 0.05) && (fabs(x[2]-0.5) <= 0.05)", L=self.problem.geometry_L)

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0, 0]), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0, 0])
		def getVolumeForce(self, V):
			return Constant([0, 0, 0])
		def getBoundaryTraction(self, V):
			return Expression(["(x[0] >= L-DOLFIN_EPS) && (fabs(x[1]-0.5) <= 0.05) && (fabs(x[2]-0.5) <= 0.05) ? 1 : 0", "0", "0"], degree=1, L=self.problem.geometry_L)

	class Loadcase2(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0, 0]), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0, 0])
		def getVolumeForce(self, V):
			return Constant([0, 0, 0])
		def getBoundaryTraction(self, V):
			return Expression(["0", "(x[0] >= L-DOLFIN_EPS) && (fabs(x[1]-0.5) <= 0.05) && (fabs(x[2]-0.5) <= 0.05) ? -1 : 0", "0"], degree=1, L=self.problem.geometry_L)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
			#self.Loadcase2(self),
		]



class PlateComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 2
		self.geometry_R = 1 - 1/2**0.5
		self.geometry_D = self.geometry_R*0.5
	def getMesh(self, R):
		if 0:
			mesh = RectangleMesh(Point(0, 0), Point(1 + 1e-8, 1), int(R), int(R))
			mesh = refine(mesh)
			return mesh
		domain = mshr.Rectangle(Point(0, 0), Point(1, 1))
		return domain
	def getInitialRho(self, V):
		rho = interpolate(Expression("(fabs(x[0]) < R) || (fabs(x[1]) < R) ? 1 : rho_min", R=self.geometry_R, rho_min=float(self.rho_min), degree=1), V)
		return rho
	def getInjectionDomain(self):
		return CompiledSubDomain("x[0] >= 1-DOLFIN_EPS || x[1] >= 1-DOLFIN_EPS")
		return CompiledSubDomain("x[0] >= 1-DOLFIN_EPS")

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V.sub(0), Constant(0), CompiledSubDomain("on_boundary && (x[0] <= DOLFIN_EPS)")),
			DirichletBC(V.sub(1), Constant(0), CompiledSubDomain("on_boundary && (x[1] <= DOLFIN_EPS)")),
			#DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			#DirichletBC(V.sub(0), Constant(0), CompiledSubDomain("on_boundary && (x[0] <= -1+DOLFIN_EPS)")),
			#DirichletBC(V.sub(0), Constant(0), CompiledSubDomain("on_boundary && (x[1] <= DOLFIN_EPS")),
			#DirichletBC(V.sub(1), Constant(0), CompiledSubDomain("on_boundary && (x[1] <= DOLFIN_EPS)")),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, 0])
		def getBoundaryTraction(self, V):
			return Expression(["((x[0] >= 1-DOLFIN_EPS) && (fabs(x[1]) <= D+DOLFIN_EPS) ? 1 : 0)", "0"], R=self.problem.geometry_R, D=self.problem.geometry_D, degree=1)
			return Expression(["(x[0] > 0) ? exp(-9*(x[1]*x[1])/(R*R)) : 0", "0"], R=self.problem.geometry_R, degree=4)
			return Expression(["((x[0] >= 1-DOLFIN_EPS) && (fabs(x[1]) < R) ? 1 : 0) + ((x[0] <= -1+DOLFIN_EPS) && (fabs(x[1]) < R) ? -1 : 0)", "0"], R=self.problem.geometry_R, degree=1)

	class Loadcase2(Loadcase1):
		def getBoundaryTraction(self, V):
			return Expression(["0", "(x[1] >= 1-DOLFIN_EPS) && (fabs(x[0]) <= D+DOLFIN_EPS) ? 4 : 0"], R=self.problem.geometry_R, D=self.problem.geometry_D, degree=1)
			return Expression(["(x[1] > 0) ? exp(-9*(x[0]*x[0])/(R*R)) : 0", "0"], R=self.problem.geometry_R, degree=4)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
			self.Loadcase2(self),
		]


class CircleComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 1
		self.geometry_R = 1
	def getMesh(self, R):
		domain = mshr.Circle(Point(0, 0), self.geometry_R, self.mesh_R)
		return domain
	def getInitialRho(self, V):
		R0 = bisect(lambda R0: R0**2/self.geometry_R**2*float(self.rho_min) + (1 - R0**2/self.geometry_R**2) - float(self.rho_bar),
			0.0, self.geometry_R)
		rho = interpolate(Expression("x[0]*x[0] + x[1]*x[1] > R0*R0 ? 1 : rho_min", R0=R0, rho_min=float(self.rho_min), degree=1), V)
		return rho
	def getInjectionDomain(self):
		return CompiledSubDomain("on_boundary")
		return CompiledSubDomain("x[0] < DOLFIN_EPS")
		return CompiledSubDomain("pow(fabs(x[0]-0.5), 2.0) + pow(fabs(x[1]-0.5), 2.0) < 0.01")
		return CompiledSubDomain("(x[0] < DOLFIN_EPS || x[0] >= 1-DOLFIN_EPS)")

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary")),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, 1])
		def getBoundaryTraction(self, V):
			return Constant([0, 0])

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]


class BoxComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 1
		self.geometry_L = 250
		self.geometry_H = 100
	def getMesh(self, R):
		#domain = mshr.Surface3D("../meshes/plate_w_inset.stl")
		domain = mshr.Rectangle(Point(0, 0), Point(self.geometry_L, self.geometry_H)) #- mshr.Circle(Point(72.5, 50.0), 15.0)
		return domain
	def _getInitialRho(self, V):
		D = bisect(lambda D: float(self.rho_bar) - (D*self.geometry_L + D*self.geometry_H - D*D)/(self.geometry_L*self.geometry_H) -
			(self.geometry_L*self.geometry_H - (D*self.geometry_L + D*self.geometry_H - D*D))/(self.geometry_L*self.geometry_H)*float(self.rho_min),
			0.0, max(self.geometry_L, self.geometry_H))
		print "D=",D
		rho = interpolate(Expression("(fabs(x[1] - 0.5*H) < 0.5*D || (L - x[0]) < D) ? 1.0 : rho_min",
			L=self.geometry_L, H=self.geometry_H, D=D, rho_min=float(self.rho_min), degree=1), V)
		return rho
	def getInjectionDomain(self):
		return CompiledSubDomain("x[0] < DOLFIN_EPS")

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
			return Constant([0, -1])
		def getVolumeForce(self, V):
			return Constant([0, 0])
		def getBoundaryTraction(self, V):
			return Expression(["(x[0] >= L-DOLFIN_EPS) ? 1 : 0", "(x[0] >= L-DOLFIN_EPS) ? -1 : 0"], L=self.problem.geometry_L, degree=1)
			return Constant([0, 0])

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]


class TreeComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 1
		self.geometry_L = 1.0
		self.geometry_H = 2*self.geometry_L
		self.geometry_S = 0.1*self.geometry_L
	def getMesh(self, R):
		domain = mshr.Rectangle(Point(0, 0), Point(self.geometry_L, self.geometry_H))
		return domain
	def getInjectionDomain(self):
		return CompiledSubDomain("x[1] <= DOLFIN_EPS")

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (fabs(x[0] - 0.5*L) <= S) && (x[1] <= DOLFIN_EPS)", L=self.problem.geometry_L, S=self.problem.geometry_S)),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, -1])
		def getBoundaryTraction(self, V):
			return Constant([0, 0])

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]



class SSBridgeComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 1
		self.geometry_L = 1
		self.geometry_H = 0.5*self.geometry_L
		self.geometry_S = 0.05*self.geometry_L
	def getMesh(self, R):
		domain = mshr.Rectangle(Point(0, 0), Point(self.geometry_L, self.geometry_H))
		return domain
	def getInjectionDomain(self):
		return CompiledSubDomain("x[1] >= H-DOLFIN_EPS", H=self.geometry_H)

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[0] <= S) && (x[1] <= DOLFIN_EPS)", L=self.problem.geometry_L, S=self.problem.geometry_S)),
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[0] >= L-S) && (x[1] <= DOLFIN_EPS)", L=self.problem.geometry_L, S=self.problem.geometry_S)),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, -1])
		def getBoundaryTraction(self, V):
			return Constant([0, 0])
			#return Expression(["0", "(fabs(x[0] - 0.5*L) <= 0.5*S) ? -1 : 0"], L=self.problem.geometry_L, S=self.problem.geometry_S, degree=1)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]


class BridgeComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 1
		self.geometry_L = 250
		self.geometry_H = 100
		self.geometry_S = 50
		self.geometry_CX = 150
	def getMesh(self, R):
		if 1:
			mesh = RectangleMesh(Point(0, 0), Point(self.geometry_L, self.geometry_H), int(R*self.geometry_L/self.geometry_H), int(R))
			#mesh = refine(mesh)
			return mesh
		#domain = mshr.Surface3D("../meshes/plate_w_inset.stl")
		T = 0.1
		domain = mshr.Rectangle(Point(0, 0), Point(self.geometry_L, self.geometry_H)) \
			+ mshr.Rectangle(Point(0, -T), Point(self.geometry_S, T)) \
			+ mshr.Rectangle(Point(-T, self.geometry_H-self.geometry_S), Point(T, self.geometry_H)) \
			+ mshr.Rectangle(Point(self.geometry_L-T, self.geometry_H-self.geometry_S), Point(self.geometry_L+T, self.geometry_H)) \
			+ mshr.Rectangle(Point(self.geometry_L - self.geometry_S, - T), Point(self.geometry_L, T)) \
			+ mshr.Rectangle(Point(self.geometry_CX - 0.5*self.geometry_S, self.geometry_H - T), Point(self.geometry_CX + 0.5*self.geometry_S, self.geometry_H + T))
		domain = mshr.Rectangle(Point(0, 0), Point(self.geometry_L, self.geometry_H))
		return domain
	def getInjectionDomain(self):
		return CompiledSubDomain("x[0] < DOLFIN_EPS")

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V.sub(0), Constant(0), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS) && (x[1] >= H-S-DOLFIN_EPS)", L=self.problem.geometry_L, S=self.problem.geometry_S, H=self.problem.geometry_H)),
			DirichletBC(V.sub(1), Constant(0), CompiledSubDomain("on_boundary && (x[0] < S+DOLFIN_EPS) && (x[1] < DOLFIN_EPS)", L=self.problem.geometry_L, S=self.problem.geometry_S)),
			DirichletBC(V.sub(1), Constant(0), CompiledSubDomain("on_boundary && (x[0] >= L-DOLFIN_EPS) && (x[1] >= H-S-DOLFIN_EPS)", L=self.problem.geometry_L, S=self.problem.geometry_S, H=self.problem.geometry_H)),
			DirichletBC(V.sub(0), Constant(0), CompiledSubDomain("on_boundary && (x[0] >= L-S-DOLFIN_EPS) && (x[1] < DOLFIN_EPS)", L=self.problem.geometry_L, S=self.problem.geometry_S)),
			#DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[0] >= L-S-DOLFIN_EPS) && (x[1] < DOLFIN_EPS)", L=self.problem.geometry_L, S=self.problem.geometry_S)),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, 0])
		def getBoundaryTraction(self, V):
			return Expression(["0", "(x[1] >= H-DOLFIN_EPS && fabs(x[0] - CX) <= 0.5*S+DOLFIN_EPS) ? -1 : 0"], H=self.problem.geometry_H, CX=self.problem.geometry_CX, S=self.problem.geometry_S, degree=1)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]


class PlateWithInsetComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.geometry_L = 250
		self.geometry_W = 100
		self.geometry_D = 10
		self.geometry_hole_x = 72.5
		self.geometry_hole_y = 0.5*self.geometry_W
		self.geometry_hole_R = 15.0
		self.mesh_R *= 1
		if 1:
			self.domain = mshr.Rectangle(Point(0, 0), Point(self.geometry_L, self.geometry_W)) - mshr.Circle(Point(self.geometry_hole_x, self.geometry_hole_y), self.geometry_hole_R)
		else:
			#domain = mshr.Surface3D("../meshes/plate_w_inset.stl")
			self.domain = mshr.Box(Point(0, 0, 0), Point(self.geometry_L, self.geometry_W, self.geometry_D)) - mshr.Cylinder(Point(self.geometry_hole_x, self.geometry_hole_y, 0), Point(self.geometry_hole_x, self.geometry_hole_y, self.geometry_D), self.geometry_hole_R, self.geometry_hole_R)

	def getMesh(self, R):
		return self.domain
	def getInjectionDomain(self):
		return CompiledSubDomain("x[0] <= DOLFIN_EPS")

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0]*self.problem.mesh_gdim), CompiledSubDomain("on_boundary && x[0] < DOLFIN_EPS", L=self.problem.geometry_L)),
			#DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && x[0] < DOLFIN_EPS || x[0] >= L-DOLFIN_EPS", L=self.problem.geometry_L)),
			#DirichletBC(V, Constant([0, 1]), CompiledSubDomain("on_boundary && (x[0]-CX)*(x[0]-CX) + (x[1]-CY)*(x[1]-CY) <= 1.001*R*R", CX=self.problem.geometry_hole_x, CY=self.problem.geometry_hole_y, R=self.problem.geometry_hole_R)),
			]
		def getVolumeGravity(self, V):
			return Constant([0]*self.problem.mesh_gdim)
		def getVolumeForce(self, V):
			return Constant([0]*self.problem.mesh_gdim)
		def getBoundaryTraction(self, V):
			return Expression(["0", "(x[0]-CX)*(x[0]-CX) + (x[1]-CY)*(x[1]-CY) <= 1.001*R*R ? -1 : (x[0] >= L-DOLFIN_EPS ? 1 : 0)"] + (["0"] if self.problem.mesh_gdim == 3 else []), CX=self.problem.geometry_hole_x, CY=self.problem.geometry_hole_y, R=self.problem.geometry_hole_R, L=self.problem.geometry_L, degree=1)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]


class TestComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
	def getMesh(self, R):
		return RectangleMesh(Point(0, 0), Point(1, 1), 1, 1)
		return UnitTriangleMesh()
	def getInjectionDomain(self):
		return CompiledSubDomain("x[0] < DOLFIN_EPS")

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("(x[1] < DOLFIN_EPS)"), method="pointwise"),
			# DirichletBC(V, Constant([0, 0]), CompiledSubDomain("(x[0] < DOLFIN_EPS) && (x[1] < DOLFIN_EPS)"), method="pointwise"),
			# DirichletBC(V, Constant([1, 0]), CompiledSubDomain("(x[0] > 0.5) && (x[1] < DOLFIN_EPS)"), method="pointwise"),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, -1])
		def getBoundaryTraction(self, V):
			return Constant([0, 0])
			return Expression(["0", "(x[1] >= 1-DOLFIN_EPS) ? -1 : 0"], degree=1)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]



class NumaPDEComplianceTopologyOptimizationProblem(ComplianceTopologyOptimizationProblem):
	def __init__(self):
		ComplianceTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 4
	def getMesh(self, R):
		domain = NumaPDEDomain2D(R)
		return domain
	def getInjectionDomain(self):
		return CompiledSubDomain("x[0] < 0.5 && x[1] < DOLFIN_EPS")
		return CompiledSubDomain("x[0] < DOLFIN_EPS")

	class Loadcase1(ComplianceTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[1] < DOLFIN_EPS) && (x[0] < 0.4 + DOLFIN_EPS)")),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, 0])
		def getBoundaryTraction(self, V):
			return Expression(["0", "(x[0] > 1.4) && (x[1] < DOLFIN_EPS) ? -1 : 0"], degree=1)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		]




def NumaPDEDomain2D(R):

	# geometry parameters
	a1  = 0.2
	b1  = 0.2
	a2  = 0.375
	b2  = 0.25

	h       = 1.0
	h1      = 0.4
	h2	= h-b2
	w	= 2*a2-2*a1
	v	= 2*a1

	n1	= max(1, int(0.5*R))
	n2	= int(1.7*n1)

	# Angle arrays
	theta1 = np.arange( (n1+1), dtype='d')/(n1+1)*np.pi
	theta2 = np.arange( (n2+1), dtype='d')/(n2+1)*np.pi

	# Midpoints
	M1x	=   w +   a1
	M1y	= h1
	M2x	= 2*w + 3*a1
	M2y	= h1
	M3x	=   w + 3*a2
	M3y	= h2
	M4x	=   w +   a2
	M4y	= h2

	# Coordinates
	x0	= [ 0, w ]
	y0 	= [ 0, 0 ]

	# First arc
	x1	= a1*np.cos(np.pi-theta1) + M1x
	y1	= b1*np.sin(np.pi-theta1) + M1y

	x2	= [ w+v, w+v, 2*w+v ]
	y2 	= [  h1,   0,     0 ]

	# Second arc
	x3	= a1*np.cos(np.pi-theta1) + M2x
	y3	= b1*np.sin(np.pi-theta1) + M2y

	x4	= [ 2*w+2*v, 2*w+2*v, 3*w+2*v ]
	y4 	= [      h1,       0,       0 ]

	# Third arc
	x5	= a2*np.cos(theta2) + M3x
	y5	= b2*np.sin(theta2) + M3y

	# Forth arc
	x6	= a2*np.cos(theta2) + M4x
	y6	= b2*np.sin(theta2) + M4y

	x7	= [  w, w, 0 ]
	y7 	= [ h2, h, h ]


	# Collect coordinates
	x = np.concatenate((x0, x1, x2, x3, x4, x5, x6, x7))

	y = np.concatenate((y0, y1, y2, y3, y4, y5, y6, y7))

	domain = mshr.Polygon([Point(x[i], y[i]) for i in range(len(x))])

	return domain
	#mesh = mshr.generate_mesh(domain, R)
	#plot(mesh, interactive=True)



class TBeamParameterIdentificationTopologyOptimizationProblem(ParameterIdentificationTopologyOptimizationProblem):
	def __init__(self):
		ParameterIdentificationTopologyOptimizationProblem.__init__(self)
		self.mesh_R *= 0.5
		self.geometry_L = 1
		self.geometry_H = 1
	def getMesh(self, R):
		if 0:
			return RectangleMesh(Point(0, 0), Point(1, 1), int(R), int(R))
		#domain = mshr.Surface3D("../meshes/plate_w_inset.stl")
		domain = mshr.Rectangle(Point(0, 0), Point(self.geometry_L, self.geometry_H))
		s = 0.5
		#domain = mshr.Rectangle(Point(0, 0), Point(self.geometry_L, self.geometry_H)) - mshr.Rectangle(Point(0, 0), Point(self.geometry_L*s, self.geometry_H*s))
		return domain
	def getInitialRho(self, V):
		rho = interpolate(Expression("x[1] < D ? 1.0 : rho_min",
			D=float(self.rho_bar), rho_min=float(self.rho_min), degree=1), V)
		return rho
	"""
		D = bisect(lambda D: float(self.rho_bar) - (D*self.geometry_L + D*self.geometry_H - D*D)/(self.geometry_L*self.geometry_H) -
			(self.geometry_L*self.geometry_H - (D*self.geometry_L + D*self.geometry_H - D*D))/(self.geometry_L*self.geometry_H)*float(self.rho_min),
			0.0, max(self.geometry_L, self.geometry_H))
		rho = interpolate(Expression("(fabs(x[1] - 0.5*H) < 0.5*D || (L - x[0]) < D) ? 1.0 : rho_min",
			L=self.geometry_L, H=self.geometry_H, D=D, rho_min=float(self.rho_min), degree=1), V)
		return rho
	"""
	def getInjectionDomain(self):
		return CompiledSubDomain("x[0] < DOLFIN_EPS")
		return CompiledSubDomain("(x[0] < DOLFIN_EPS || x[0] >= L-DOLFIN_EPS)", L=self.geometry_L)

	def getDirichletBC(self, V):
		return [
		#DirichletBC(V, Constant(1), CompiledSubDomain("x[1] < 0.5")),
		#DirichletBC(V, Constant(1), CompiledSubDomain("1-x[1] < (0.1 + 0.2*sin(0.5*3.1415*x[0]))")),
		]

	class Loadcase1(ParameterIdentificationTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V.sub(0), Constant(0), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			#DirichletBC(V.sub(0), Constant(1), CompiledSubDomain("on_boundary && (x[0] >= L-DOLFIN_EPS)", L=self.problem.geometry_L)),
			DirichletBC(V, Constant([1,0]), CompiledSubDomain("on_boundary && (x[0] >= L-DOLFIN_EPS)", L=self.problem.geometry_L)),
			DirichletBC(V.sub(1), Constant(0), CompiledSubDomain("on_boundary && (x[1] < DOLFIN_EPS)")),
			]
			return [
			DirichletBC(V, Constant([0,0]), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			DirichletBC(V.sub(1), Constant(1), CompiledSubDomain("on_boundary && (x[0] >= L-DOLFIN_EPS)", L=self.problem.geometry_L)),
			]
			return [
			DirichletBC(V, Constant([0,0]), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			]
			return [
			DirichletBC(V, Constant([0,0]), CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")),
			DirichletBC(V, Constant([1,0]), CompiledSubDomain("on_boundary && (x[0] >= L-DOLFIN_EPS)", L=self.problem.geometry_L)),
			#DirichletBC(V.sub(1), Constant(0), CompiledSubDomain("on_boundary && (x[1] < DOLFIN_EPS)")),
			]
		def getVolumeGravity(self, V):
			return Constant([0, 0])
		def getVolumeForce(self, V):
			return Constant([0, 0])
		def getBoundaryTraction(self, V):
			return Constant([0, 0])
			return Expression(["(x[0] >= L-DOLFIN_EPS) ? 1 : 0", "0"], L=self.problem.geometry_L, degree=1)

	class Loadcase2(ParameterIdentificationTopologyOptimizationLoadcase):
		def getDirichletBC(self, V):
			return [
			DirichletBC(V, Constant([0, 0]), CompiledSubDomain("on_boundary && (x[1] < DOLFIN_EPS)")),
			]
		def getVolumeGravity(self, V):
			return Constant([0, -1])
		def getVolumeForce(self, V):
			return Constant([0, 0])
		def getBoundaryTraction(self, V):
			return Constant([0, 0])
			return Expression(["0", "(x[1] >= H-DOLFIN_EPS) ? 1 : 0"], H=self.problem.geometry_H, degree=1)

	def getLoadcases(self):
		return [
			self.Loadcase1(self),
		#	self.Loadcase2(self),
		]



class TopologyOptimizationApp(QtGui.QApplication):

	def __init__(self, args):
		
		QtGui.QApplication.__init__(self, args)

		print "command line:", " ".join(args)

		self.mpi_comm = mpi_comm_world() 
		self.mpi_rank = MPI.rank(self.mpi_comm)

		clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
		material_models = []
		algorithms = []
		optimality_criterions = []
		for name, cls in clsmembers:
			if issubclass(cls, MaterialModel) and cls != MaterialModel:
				material_models.append(name[0:-len("MaterialModel")])
			if issubclass(cls, TopologyOptimizationAlgorithm) and cls != TopologyOptimizationAlgorithm:
				algorithms.append(name[0:-len("TopologyOptimizationAlgorithm")])
			if issubclass(cls, OptimalityCriterion) and cls != OptimalityCriterion:
				optimality_criterions.append(name[0:-len("OptimalityCriterion")])

		log_levels = {"DBG": 10, "TRACE": 13, "PROGRESS": 16, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50, "NONE": False}

		# read parameters
		self.parameters = Parameters()
		self.parameters.add("problem", "Tip")
		self.parameters.add("optimality_criterion", "A", optimality_criterions)
		self.parameters.add("eikonal_solver", "external" if have_eikonal2d else "internal", ["internal", "external"])
		self.parameters.add("objective", "Compliance", ["Compliance", "ParameterIdentification"])
		self.parameters.add("algorithm", "Boosted", algorithms)
		self.parameters.add("material_model", "TransverseIsotropic", material_models)
		self.parameters.add("result_dir", "")
		self.parameters.add("rho_range", "continuous", ["continuous", "discrete"])
		self.parameters.add("rho_space", "CG1", ["DG0", "CG1"])
		self.parameters.add("rho_init", "auto", ["auto", "constant", "random"])
		self.parameters.add("u_space", "CG2", ["CG1", "CG2"])
		self.parameters.add("gradient_scaling", "auto", ["auto", "lumped", "mass_H1", "mass_L2", "smooth", "smooth_DG", "none"])
		self.parameters.add("renormalization", "additive", ["additive", "scaling"])
		self.parameters.add("run", 0)
		self.parameters.add("exit", 0)
		self.parameters.add("initial_refinements", 0)
		self.parameters.add("mesh_resolution", 20.0)
		self.parameters.add("eikonal_dual", False)
		self.parameters.add("tol", 1e-4)
		self.parameters.add("eta", 3.0)
		self.parameters.add("gamma", 1.0)
		self.parameters.add("theta_scale", 1.0/3.0)
		self.parameters.add("tau_scale", 1.0)
		self.parameters.add("density", "")
		self.parameters.add("alpha_max", 100.0)
		self.parameters.add("alpha_conv", 100000.0)
		self.parameters.add("alpha_relax", 1.0)
		self.parameters.add("rho_bar", 0.5)
		self.parameters.add("rho_min", 0.01)
		self.parameters.add("rho0", 0.0)
		self.parameters.add("alpha_scale", 1.0)
		self.parameters.add("alpha", 1.0)
		self.parameters.add("delta_rho", 0.05)
		self.parameters.add("penalty", 0.0)
		self.parameters.add("injection_domain", "")
		self.parameters.add("injection_domain_method", "topological", ["topological", "geometric", "pointwise"])
		self.parameters.add("ip_grad_scale", 1.0)
		self.parameters.add("anisotropy_scale", 1.0)
		self.parameters.add("fo_angle", 0.0)
		self.parameters.add("tau", 0.9)
		self.parameters.add("dismiss", False)
		self.parameters.add("E", 1.0)
		self.parameters.add("nu", 0.3)
		self.parameters.add("quadrature_degree", -1)
		self.parameters.add("num_threads", 0)
		self.parameters.add("reorder_dofs", False)	# TODO: fiber orientation plots shows wrong data, due to sorting
		self.parameters.add("isotropy_degree", 0.0)
		self.parameters.add("gradient_terms", "auto", ["auto", "all", "isotropic", "anisotropic"])
		self.parameters.add("objective_form", "penalized_compliance", ["penalized_compliance", "compliance"])
		self.parameters.add("log_level", "INFO", log_levels.keys())
		self.parameters.add("num_adaptions", 0)
		self.parameters.add("fo_file", "OpenFOAM/Tip_OpenFOAM.vtk")
		self.parameters.add("num_steps", 100)
		self.parameters.add("num_refinements", 0)
		self.parameters.add("write_interval", 10)

		"""
		mm = Parameters()
		mm.add("material_model", self.parameters["material_model"])
		mm.parse([a for a in args if a.startswith("--material_model=")])

		if mm["material_model"] == "TransverseIsotropic":
			pass
		else:
			pass
		"""

		self.parameters.parse(args)

		if self.parameters["eikonal_solver"] == "external":
			if self.parameters["gradient_terms"] == "auto":
				self.parameters["gradient_terms"] = "isotropic"		# TODO
		elif self.parameters["eikonal_solver"] == "internal":
			if self.parameters["gradient_terms"] == "auto":
				self.parameters["gradient_terms"] = "all"

		if self.parameters["u_space"] == "CG2":
			u_degree = 2
		else:
			u_degree = 1

		if self.parameters["rho_space"] == "DG0":
			if self.parameters["gradient_scaling"] == "auto":
				self.parameters["gradient_scaling"] = "smooth"
			if self.parameters["quadrature_degree"] < 0:
				self.parameters["quadrature_degree"] = 2
			self.parameters["reorder_dofs"] = False
		elif self.parameters["rho_space"] == "CG1":
			if self.parameters["gradient_scaling"] == "auto":
				self.parameters["gradient_scaling"] = "mass_H1"
			if self.parameters["quadrature_degree"] < 0:
				self.parameters["quadrature_degree"] = 5	# For the adjoint equation we need 5 for eta=3 and CG2 elements for u


		#list_lu_solver_methods()

		# TODO: this should be something like ceil(eta)
		parameters["form_compiler"]["quadrature_degree"] = int(self.parameters["quadrature_degree"])
		parameters["reorder_dofs_serial"] = self.parameters["reorder_dofs"]
		parameters['allow_extrapolation'] = True
		parameters["num_threads"] = int(self.parameters["num_threads"])
		parameters["mesh_partitioner"] = "ParMETIS"
		#parameters["linear_algebra_backend"] = "Eigen"

		#parameters["form_compiler"]["representation"] = "uflacs"

		# set the log level
		ll = log_levels[self.parameters["log_level"]]
		if ll > 0:
			set_log_level(ll)
		else:
			set_log_active(False)

		self.settings = QtCore.QSettings("NumaPDE", "TopologyOptimization")
		self.window = TopologyOptimizationMainWindow()

	def notify(self, receiver, event):
		try:
			QtGui.QApplication.notify(self, receiver, event)
		except:
			QtGui.QMessageBox.critical(self, "Error", sys.exc_info()[0])
		return False

	def getSettingByteArray(self, key):
		s = self.settings.value(key)
		if hasattr(s, "toByteArray"):
			return s.toByteArray()
		return s

	def restoreWindowState(self, win, prefix):
		win.restoreGeometry(self.getSettingByteArray(prefix + "_geometry"))
		if (isinstance(win, QtGui.QMainWindow)):
			win.restoreState(self.getSettingByteArray(prefix + "_windowState"))

	def saveWindowState(self, win, prefix):
		self.settings.setValue(prefix + "_geometry", win.saveGeometry())
		if (isinstance(win, QtGui.QMainWindow)):
			self.settings.setValue(prefix + "_windowState", win.saveState())

class FunctionPlot(QtGui.QFrame):
	
	def __init__(self, gdim, func, label, vmin=None, vmax=None, cmap='Greys', color=None, save=True, parent=None):

		self.gdim = gdim
		self.func = func if isinstance(func, list) else [func]
		self.label = label
		self.show_grid = [False]*len(func) if isinstance(func, list) else [False]
		self.mesh_id = [0]*len(func) if isinstance(func, list) else [0]
		self.vmin = vmin if isinstance(vmin, list) else [vmin]
		self.vmax = vmax if isinstance(vmax, list) else [vmax]
		self.cmap = cmap if isinstance(cmap, list) else [cmap]
		self.color = color if isinstance(color, list) else [color]
		self.dirty = True
		self.save = save

		self.cb = [None]*len(self.func)
		self.grid = [None]*len(self.func)
		self.tri = [None]*len(self.func)

		QtGui.QFrame.__init__(self, parent) 

		self.fig = Figure(figsize=(20,20))
		self.fig.set_tight_layout(None)
		self.fig.set_frameon(False)

		self.figcanvas = FigureCanvas(self.fig)
		self.figcanvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

		self.fignavbar = NavigationToolbar(self.figcanvas, self)
		self.fignavbar.set_cursor(cursors.SELECT_REGION)


		if self.gdim == 1:
			self.axes = self.fig.add_subplot(111)
		elif self.gdim == 2:
			self.axes = self.fig.add_subplot(111)
			self.axes.set_axis_bgcolor('#aaaaff')
		else:
			self.axes = self.fig.add_subplot(111, projection='3d')
			self.axes.grid(False)
			self.axes.set_aspect('equal') 


		#self.setFrameStyle(QtGui.QFrame.StyledPanel)
		self.setMinimumSize(300, 300)
		self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
		self.setLineWidth(1)
		#self.setContentsMargins(3, 3, 3, 3)

		"""
		self.frame = QtGui.QFrame()
		fvbox = QtGui.QVBoxLayout(self.frame)
		"""

		vbox = QtGui.QVBoxLayout()
		vbox.setContentsMargins(0, 0, 0, 0)
		vbox.addWidget(self.fignavbar)
		vbox.addWidget(self.figcanvas)
		self.setLayout(vbox)

	def plot_1d(self, f, index):

		if index == 0:
			self.axes.clear()

		f(self.axes)
		
	def plot_2d(self, f, index):
		
		V = f.function_space()
		mesh = V.mesh()
		mesh_id = id(mesh)

		sig = V.dolfin_element().signature()
		is_DG = "Discontinuous Lagrange" in sig
		if is_DG:
			Cf = f.vector().array()
			C = Cf
		else:
			C = f.compute_vertex_values(mesh)
			Cf = None

		if (mesh_id == self.mesh_id[index]):
			# quick update
			if not self.tri[index] is None:
				self.tri[index].set_array(C)
			if not self.grid[index] is None:
				self.grid[index].set_visible(self.show_grid[index])
			return

		"""
		ntri = self.mesh
		for c in entities(self.mesh, 0):
			for v in entities(c, 1):
				triangles[
		"""

		cbax = None
		if not self.cb[index] is None:
			cbax = self.cb[index].ax
			cbax.clear()

		if index == 0:
			self.axes.clear()

		triangles = mesh.cells()
		coordinates = mesh.coordinates()

		# vertex coordinates
		x = coordinates[:, 0]
		y = coordinates[:, 1]

		# cell centers
		xc = np.mean(x[triangles], axis=1)
		yc = np.mean(y[triangles], axis=1)

		rank = max(1, V.num_sub_spaces())

		if rank == 1:
			
			vmin = self.vmin[index]
			vmax = self.vmax[index]
			"""
			if vmin is None or vmax is None:
				mC = np.mean(C)
				sC = np.std(C)
				alpha = 2.0
				if vmin is None:
					vmin = mC - alpha*sC
				if vmax is None:
					vmax = mC + alpha*sC
			"""
			if vmin is None or vmax is None:
				mC = np.mean(C)
				alpha = 2.0
				if vmin is None:
					Cn = C[np.where(C < mC)]
					sC = np.std(Cn)
					mC = np.mean(Cn)
					vmin = mC - alpha*sC

				if vmax is None:
					Cn = C[np.where(C > mC)]
					sC = np.std(Cn)
					mC = np.mean(Cn)
					vmax = mC + alpha*sC

			# scalar plot
			self.tri[index] = mtri.Triangulation(x, y, triangles=triangles)
			plot = self.axes.tripcolor(self.tri[index], C, facecolors=Cf, shading=('gouraud' if Cf is None else 'flat'), cmap=matplotlib.cm.get_cmap(self.cmap[index]), vmin=vmin, vmax=vmax)

			# add colorbar
			if cbax is None:
				self.cb[index] = self.fig.colorbar(plot, shrink=1.0, pad=0.02, fraction=0.02)
			else:
				self.cb[index] = self.fig.colorbar(plot, cax=cbax)


			#print "after:", self.axes.get_children()
			dummy = self.axes.triplot(self.tri[index], alpha=1, color="blue")
			#print "after:", self.axes.get_children()
			
			if dummy:
				self.grid[index], dummy = dummy
			else:
				for c in self.axes.get_children():
					#if isinstance(c, matplotlib.collections.TriMesh):
					#if isinstance(c, matplotlib.lines.Line2D):
					if isinstance(c, matplotlib.patches.PathPatch):
						self.grid[index] = c
						break

			self.grid[index].set_visible(self.show_grid[index])

		elif rank == 2 and is_DG:
			
			# vector plot
			C = C.reshape(2, xc.shape[0])
			u = C[0,:]
			v = C[1,:]

			quiveropts = dict(headaxislength=0, headlength=0, pivot='middle', headwidth=1, width=0.001, linewidth=0, units="xy", scale=None)

			if self.cmap[index] is None:
				plot = self.axes.quiver(xc, yc, u, v, color=self.color[index], **quiveropts)
			else:
				cm = matplotlib.cm.get_cmap(self.cmap[index])
				color = np.sqrt(u**2 + v**2)
				nz = matplotlib.colors.Normalize()
				nz.autoscale(color)
				plot = self.axes.quiver(xc, yc, u, v, color=cm(nz(color)), **quiveropts)

				m = matplotlib.cm.ScalarMappable(cmap=cm, norm=nz)
				m.set_array(color)

				# add colorbar
				if cbax is None:
					self.cb[index] = self.fig.colorbar(m, shrink=1.0, pad=0.02, fraction=0.02)
				else:
					self.cb[index] = self.fig.colorbar(m, cax=cbax)

		elif rank == 2 and not is_DG:
			
			# vector plot
			C = C.reshape(2, coordinates.shape[0])
			u = C[0,:]
			v = C[1,:]

			quiveropts = dict(headaxislength=0, headlength=0, pivot='middle', headwidth=1, width=0.001, linewidth=0, units="xy", scale=None)

			if self.cmap[index] is None:
				plot = self.axes.quiver(x, y, u, v, color=self.color[index], **quiveropts)
			else:
				cm = matplotlib.cm.get_cmap(self.cmap[index])
				color = np.sqrt(u**2 + v**2)
				nz = matplotlib.colors.Normalize()
				nz.autoscale(color)
				plot = self.axes.quiver(x, y, u, v, color=cm(nz(color)), **quiveropts)

				m = matplotlib.cm.ScalarMappable(cmap=cm, norm=nz)
				m.set_array(color)

				# add colorbar
				if cbax is None:
					self.cb[index] = self.fig.colorbar(m, shrink=1.0, pad=0.02, fraction=0.02)
				else:
					self.cb[index] = self.fig.colorbar(m, cax=cbax)
		else:
			pass


		self.axes.set_xlim([np.min(x),np.max(x)])
		self.axes.set_ylim([np.min(y),np.max(y)])

		self.mesh_id[index] = mesh_id

	def plot_3d(self, f, index):
		#self.axes.plot_trisurf(x, y, z, facecolors=color, triangles=triangles)

		if index == 0:
			self.axes.clear()

		V = f.function_space()
		mesh = V.mesh()
		mesh_id = id(mesh)
		C = f.compute_vertex_values(mesh)

		if (mesh_id == self.mesh_id[index]):
			# quick update
			#self.tri[index].set_array(C)
			#self.grid[index].set_visible(self.show_grid[index])
			#return
			pass
		
		cells = mesh.cells()
		n = cells.shape[0]
		triangles = np.zeros((4*n, 3), dtype=np.int)

		triangles[0*n:1*n, 0] = cells[:,0]
		triangles[0*n:1*n, 1] = cells[:,1]
		triangles[0*n:1*n, 2] = cells[:,2]

		triangles[1*n:2*n, 0] = cells[:,0]
		triangles[1*n:2*n, 1] = cells[:,1]
		triangles[1*n:2*n, 2] = cells[:,3]

		triangles[2*n:3*n, 0] = cells[:,0]
		triangles[2*n:3*n, 1] = cells[:,2]
		triangles[2*n:3*n, 2] = cells[:,3]

		triangles[3*n:4*n, 0] = cells[:,1]
		triangles[3*n:4*n, 1] = cells[:,2]
		triangles[3*n:4*n, 2] = cells[:,3]

		# remove duplicate triangles
		triangles.sort(axis=1)
		b = np.ascontiguousarray(triangles).view(np.dtype((np.void, triangles.dtype.itemsize * triangles.shape[1])))
		dummy, idx = np.unique(b, return_index=True)
		triangles = triangles[idx]
		n = triangles.shape[0]

		#x = coordinates[:, 0]
		#y = coordinates[:, 1]
		#z = coordinates[:, 2]


		"""
		# treshold triangles
		treshold = 0.5
		Cmean = np.mean(C[triangles], axis=1)
		triangles = triangles[np.nonzero(Cmean > treshold)]
		n = triangles.shape[0]
		"""

		if n == 0:
			return

		if 1:
			colors = np.zeros((n, 4), dtype=np.double)
			colors[:,3] = (n**(-1.0/3.0))*np.minimum(1, np.maximum(0, np.mean(C[triangles], axis=1)))
		else:
			colors = np.zeros((n, 3), dtype=np.double)

		coordinates = mesh.coordinates()
		#poly3d = [coordinates[triangles[ix][:], :] for ix in range(len(triangles))]
		poly3d = coordinates[triangles, :]

		x = coordinates[:, 0]
		y = coordinates[:, 1]
		z = coordinates[:, 2]

		#self.axes.scatter(x,y,z)
		self.tri[index] = Poly3DCollection(poly3d, linewidths=0.5 if self.show_grid[index] else 0.0)
		self.tri[index].set_facecolors(colors)
		self.axes.add_collection3d(self.tri[index])

		self.axes.set_xlim([np.min(x),np.max(x)])
		self.axes.set_ylim([np.min(y),np.max(y)])
		self.axes.set_zlim([np.min(z),np.max(z)])

		self.mesh_id[index] = mesh_id

	def plot(self):
	
		for index, f in enumerate(self.func):
			if self.gdim == 1:
				self.plot_1d(f, index)
			elif self.gdim == 2:
				self.plot_2d(f(), index)
			elif self.gdim == 3:
				self.plot_3d(f(), index)

		self.figcanvas.draw()
		self.dirty = False


class ParamRangeControl(QtGui.QWidget):
	def __init__(self, params, parent=None): 

		QtGui.QWidget.__init__(self, parent) 

		self.params = params
		self.scale = 10**(params.digits+1)

		self.slider = QtGui.QSlider(parent=self)
		self.slider.setOrientation(QtCore.Qt.Horizontal)
		self.slider.setMinimum(int(params.min*self.scale))
		self.slider.setMaximum(int(params.max*self.scale))
		self.slider.setValue(int(params.getValue()*self.scale))
		#self.slider.setTickPosition(QtGui.QSlider.TicksBothSides)
		#self.slider.sliderMoved.connect(self.sliceSliderChanged)
		self.slider.setTickInterval(self.scale)
		self.label = QtGui.QLabel(params.name, parent=self)
		self.label.setMinimumSize(80, self.label.height())

		self.text = QtGui.QLineEdit(parent=self)
		validator = QtGui.QDoubleValidator(params.min, params.max, params.digits)
		self.text.setValidator(validator)
		self.text.setMaximumSize(100, self.text.height())

		self.slider.valueChanged.connect(self.valueChanged)
		self.text.textChanged.connect(self.textChanged)
		self.valueChanged()

		# create layout
		hbox = QtGui.QHBoxLayout()
		hbox.setContentsMargins(0, 0, 0, 0)
		hbox.addWidget(self.label)
		hbox.addWidget(self.slider)
		hbox.addWidget(self.text)
		self.setLayout(hbox)

	def textChanged(self):
		try:
			value = float(str(self.text.text()))
		except:
			return
		self.slider.setValue(value*self.scale)

	def valueChanged(self):
		value = self.slider.value()/float(self.scale)
		print self.label.text(), "set to", value
		self.params.setValue(value)
		self.params.valueChanged()
		self.text.setText(("%%.%df" % self.params.digits) % value)



class ParamEdit(QtGui.QWidget):
	def __init__(self, params, parent=None): 

		QtGui.QWidget.__init__(self, parent) 

		controls = []
		for param in params:
			if param.type == "range":
				control = ParamRangeControl(param)
				#control.setMaximumSize(300, control.height())
				#control.setMinimumSize(200, control.height())
				controls.append(control)

		# create layout
		vbox = QtGui.QVBoxLayout()
		vbox.setContentsMargins(0, 0, 0, 0)
		for control in controls:
			vbox.addWidget(control)
		self.setLayout(vbox)


def createInstance(class_name, base_type, *args):
	g = globals()
	class_name = class_name + base_type.__name__
	if not class_name in g:
		raise RuntimeError("class %s not found" % class_name)
	class_type = g[class_name]
	instance = class_type(*args)
	if not isinstance(instance, base_type):
		raise RuntimeError("class %s is not a %s" % (class_name, base_type))
	return instance


class OptimizationStepData(object):

	def __init__(self):
		self.objective = None
		self.gradient_norm = None
		self.projected_gradient_norm = None
		self.alpha = None
		self.alpha_relax = None


class TopologyOptimizationControl(QtGui.QWidget): 

	def __init__(self, parent=None): 
		 
		QtGui.QWidget.__init__(self, parent) 

		app = QtGui.QApplication.instance()

		self.step_data = []

		self.problem = createInstance(app.parameters["problem"] + app.parameters["objective"], TopologyOptimizationProblem)
		self.problem.init()

		self.algorithm = createInstance(app.parameters["algorithm"], TopologyOptimizationAlgorithm, self.problem)
		
	
		self.function_plots = []
		self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, lambda: self.problem.rho, "density", 0.0, 1.0))
		
		if not isinstance(self.algorithm, DiscreteSetTopologyOptimizationAlgorithm):
			self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, self.get_rho_scaled, "scaled density", 0.0, 1.0, save=False))

		self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, lambda: self.problem.dF, "gradient", save=False, cmap='Greys_r'))
		self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, self.get_projected_dF, "projected gradient", save=False, cmap='Greys_r'))


		def makeFun(f, *args):
			return lambda: f(*args)

		for i, lc in enumerate(self.problem.loadcases):
			#self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, makeFun(self.get_boundary_tractions, i), "f_%d" % i, cmap='jet', save=False))
			self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, makeFun(self.get_displacement, i), "displacement_%d" % i, cmap='jet', save=False))
			self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, makeFun(self.get_stress, i), "stress_%d" % i, 0.0, cmap='jet'))
			self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, makeFun(self.get_mieses, i), "mieses_%d" % i, 0.0, cmap='jet'))

			"""
			for k in range(2):
				for l in range(k, 2):
					self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, makeFun(self.get_strain_comp, i, k, l), "strain_%d_%d_%d" % (i,k,l), cmap='jet', save=False))
			"""

			#self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, makeFun(self.get_strain, i), "strain_%d" % i, 0.0, cmap='jet', save=False))
			#self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, makeFun(self.get_compliance, i), "compliance_%d" % i, 0.0, cmap='jet', save=False))

		if isinstance(self.problem.material_model, TransverseIsotropicMaterialModel):
			#self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, [lambda: self.problem.eikonal_p], "fiber orientation", cmap=[None], save=False))
			self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, [lambda: self.problem.rho, lambda: self.problem.eikonal_p], "fiber orientation", [0.0, None], [1.0, None], cmap=['Greys', None], color=[None, 'r'], save=False))
			#self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, lambda: self.problem.eikonal_p[0], "fiber orientation", 0.0, 1.0, save=False))
			self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, lambda: self.problem.eikonal, "fill time", 0.0, save=False))
			self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, lambda: self.problem.eikonal_adj, "adjoint eik", save=False))
			self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, self.get_eikonal_deriv, "eikonal_deriv", save=False))
			#self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, makeFun(self.get_eikonal_adj_rhs), "eik adj rhs", 0.0, cmap='jet', save=False))
			#self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, makeFun(self.get_eikonal_grad_norm), "inv eik grad norm", 0.0, cmap='jet', save=False))
			#self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, makeFun(self.get_eikonal_angle), "eik angle", cmap='jet', save=False))
			#self.function_plots.append(FunctionPlot(self.problem.mesh_gdim, self.get_grad_phi_norm, "grad phi norm", 0.0, None, save=False))

		self.function_plots.append(FunctionPlot(1, self.get_objective_plot, "objective", save=False))
		#self.function_plots.append(FunctionPlot(1, self.get_gradient_norm_plot, "grad norm", save=False))
		self.function_plots.append(FunctionPlot(1, self.get_proj_gradient_norm_plot, "proj grad norm", save=False))
		self.function_plots.append(FunctionPlot(1, self.get_alpha_plot, "step length", save=False))
		self.function_plots.append(FunctionPlot(1, self.get_alpha_relax_plot, "alpha relax", save=False))
		self.function_plots.append(FunctionPlot(1, self.get_linesearch_plot, "linesearch", save=False))
		self.function_plots.append(FunctionPlot(1, self.get_fd_plot, "fd_plot", save=False))

		self.function_plot = self.function_plots[0]


		params = self.problem.params + self.algorithm.params
		for param in params:
			param.valueChanged = self.paramChanged

		self.param_edit = ParamEdit(params)


		self.buttons = []
		for label, func in [("&Step", self.step), ("&Fixed Step", self.fixed_step), ("R&eset", self.reset), ("&K-Adapt", self.kadapt), ("&Adapt", self.adapt), ("Re-A&dapt", self.readapt), ("&Refine", self.refine), ("&Coarsen", self.coarsen), ("Save &Data...", self.saveData), ("Check Adjoint", self.check_adjoint), ("Auto Run", self.autoRun), ("Save Plot...", self.savePlotPDF), ("Save &Density...", self.saveDensity), ("Load &Density...", self.loadDensity), ("Refresh", self.replot), ("List Timings", self.list_timings), ("Clear Steps", self.clear_steps)]:
			button = QtGui.QPushButton(label, self)
			def make_button_click_func(button, func):
				return lambda: self.button_click(button, func)
			button.clicked.connect(make_button_click_func(button, func))
			self.buttons.append(button)

		self.tbuttons = []
		for label, func_set, func_get in [("Show Mesh", self.show_mesh, lambda: self.function_plot.show_grid[0])]:
			button = QtGui.QPushButton(label, self)
			def make_button_toggle_func(button, func_set, func_get):
				return lambda: self.button_toggle(button, func_set, func_get)
			button.setCheckable(True)
			button.setChecked(func_get())
			button.toggled.connect(make_button_toggle_func(button, func_set, func_get))
			button.func_get = func_get
			self.tbuttons.append(button)


		self.tabWidget = QtGui.QTabWidget()
		self.tabWidget.setTabsClosable(False)
		for fp in self.function_plots:
			self.tabWidget.addTab(fp, fp.label)
		self.tabWidget.currentChanged.connect(self.tabChanged)



		# create layout
		vbox = QtGui.QVBoxLayout()
		#vbox.setContentsMargins(0, 0, 0, 0)
		vbox.addWidget(self.tabWidget)
		hbox1 = QtGui.QHBoxLayout()
		hbox2 = QtGui.QHBoxLayout()
		for i, button in enumerate(self.buttons):
			if i < 8:
				hbox1.addWidget(button)
			else:
				hbox2.addWidget(button)
		for tbutton in self.tbuttons:
			hbox2.addWidget(tbutton)
		vbox.addLayout(hbox1)
		vbox.addLayout(hbox2)
		vbox.addWidget(self.param_edit)
		self.setLayout(vbox)

		self.reset2()

	def get_grad_phi_norm(self):
		f = project(sqrt(dot(grad(self.problem.eikonal), grad(self.problem.eikonal))), self.problem.V_DG0)
		return f

	def get_displacement(self, lc):
		return self.problem.loadcases[lc].u

	def get_boundary_tractions(self, lc):
		return self.problem.loadcases[lc].f

	def get_step_plot(self, ax, attr, label):
		
		x = []
		y = []
		for i, sd in enumerate(self.step_data):
			x.append(i)
			y.append(getattr(sd, attr))
			
		ax.plot(x, y, marker="o")
		ax.set_xlabel("step")
		ax.set_ylabel(label)
		ax.grid()


	def get_objective_plot(self, ax):
		self.get_step_plot(ax, "objective", "objective")
		ax.set_yscale("log")

	def get_gradient_norm_plot(self, ax):
		self.get_step_plot(ax, "gradient_norm", "gradient norm")

	def get_proj_gradient_norm_plot(self, ax):
		self.get_step_plot(ax, "projected_gradient_norm", "projected gradient norm")
		ax.set_yscale("log")

	def get_alpha_plot(self, ax):
		self.get_step_plot(ax, "alpha", "alpha")

	def get_alpha_relax_plot(self, ax):
		self.get_step_plot(ax, "alpha_relax", "alpha relax")

	def get_linesearch_plot(self, ax):
		
		dF = self.problem.dF.vector().array()
		rho = self.problem.rho.vector()
		rho_old = np.array(rho.array())

		def eval_F_alpha(alpha):
			rho[:] = rho_old - alpha*dF
			self.problem.projectDensity()
			self.problem.computeForward()
			F_alpha = self.problem.computeObjective()
			return F_alpha
		
		x = np.logspace(-1, 3, 50, base=10)
		x = np.linspace(323, 390, 20)
		x = np.logspace(-5, 3, 16, base=10)

		y = []
		for alpha in x:
			print "alpha =", alpha
			y.append(eval_F_alpha(alpha))
		
		ax.plot(x, y, marker="o")
		ax.set_xscale("log", basex=10)
		ax.set_yscale("log", basex=10)
		ax.set_xlabel(r"\alpha")
		ax.set_ylabel(r"objective")
		ax.grid()

		eval_F_alpha(0.0)

	def get_fd_plot(self, ax):
		
		h_list, dF_list, dF_fd_list = self.problem.checkAdjoint(1)

		ax.plot(np.array(h_list), np.array(dF_fd_list) - np.array(dF_list), marker="o")
		ax.set_xscale("log", basex=10)
		ax.set_yscale("log", basex=10)
		ax.set_xlabel(r"\delta")
		ax.set_ylabel(r"error")
		ax.grid()
	
	def get_stress(self, lc):
		app = QtGui.QApplication.instance()
		u_space = app.parameters["u_space"]
		u = self.get_displacement(lc)
		sigma = self.problem.sigma(eps(u))
		f = project((self.problem.rho**self.problem.eta)*sqrt(inner(sigma, sigma)), self.problem.V_DG0)
		#f = project(sqrt(inner(sigma, sigma)), self.problem.V_DG0)
		#plot(f, interactive=True)
		return f

	def get_eikonal_deriv(self):
		index = 0
		delta = 1e-5
		eikonal0 = self.problem.eikonal.copy(deepcopy=True)
		rho_old = self.problem.rho.vector()[index][0]
		self.problem.rho.vector()[index] = rho_old + delta
		self.problem.computeForward()
		eikonal0.vector().axpy(-1.0, self.problem.eikonal.vector())
		self.problem.rho.vector()[index] = rho_old
		return eikonal0

	def get_mieses(self, lc):
		app = QtGui.QApplication.instance()
		u_space = app.parameters["u_space"]
		u = self.get_displacement(lc)
		n = u.geometric_dimension()
		sigma = self.problem.sigma(eps(u))
		sigma = sigma - tr(sigma)*Identity(n)/n
		f = project((self.problem.rho**self.problem.eta)*sqrt(3.0/2.0*inner(sigma, sigma)), self.problem.V_DG0)
		#f = project(sqrt(inner(sigma, sigma)), self.problem.V_DG0)
		#plot(f, interactive=True)
		return f

	def get_strain(self, lc):
		app = QtGui.QApplication.instance()
		u_space = app.parameters["u_space"]
		u = self.get_displacement(lc)
		strain = eps(u)
		f = project((self.problem.rho**self.problem.eta)*sqrt(inner(strain, strain)), self.problem.V_DG0)
		return f

	def get_strain_comp(self, lc, i, j):
		app = QtGui.QApplication.instance()
		u = self.get_displacement(lc)
		strain = eps(u)
		f = project((self.problem.rho**self.problem.eta)*strain[i,j], self.problem.V_DG0)
		return f

	def get_eikonal_adj_rhs(self):
		f = project(self.problem.eikonal_theta*(self.problem.rho**(self.problem.eikonal_theta-1))*self.problem.eikonal_adj, self.problem.V_eikonal)
		return f

		v = assemble(self.problem.form_eikonal_adj_l)
		f = Function(self.problem.V_eikonal)
		f.vector()[:] = v[:]
		return f

	def get_eikonal_grad_norm(self):
		f = project(1/self.problem.eikonal_grad_norm, self.problem.V_DG0)
		return f

	def get_eikonal_angle(self):
		f = project(atan(self.problem.eikonal_p[1]/self.problem.eikonal_p[0]), self.problem.V_DG0)
		return f

	def get_compliance(self, lc):
		TODO
		app = QtGui.QApplication.instance()
		u_space = app.parameters["u_space"]
		lc = self.problem.loadcases[lc]
		inner(lc.f, lc.u)*phi*ds_ + inner(lc.g0 + rho*lc.g1, lc.u)*dx_
		return f

	def get_projected_dF(self):
		alpha = float(self.problem.alpha)
		rho_old = self.problem.rho
		rho = self.problem.rho = self.problem.rho.copy(deepcopy=True)
		rho.vector().axpy(-alpha, self.problem.dF.vector())
		self.problem.projectDensity()
		self.problem.rho = rho_old
		rho.vector()[:] = (rho_old.vector().array() - rho.vector().array())/alpha
		#print "func mean=", assemble(rho*dx)/self.problem.mesh_V
		return rho

	def get_rho_scaled(self):
		f = Function(self.problem.rho.function_space())
		f.vector()[:] = self.problem.rho.vector().array()**float(self.problem.eta)
		return f

	def tabChanged(self, index):
		self.function_plot = self.function_plots[index]
		if self.function_plot.dirty:
			self.function_plot.plot()
		for tbutton in self.tbuttons:
			tbutton.setChecked(tbutton.func_get())

	def paramChanged(self):
		pass

	def button_click(self, button, func):
		button.setEnabled(False)
		QtGui.QApplication.processEvents()
		try:
			func()
		except:
			traceback.print_exc()
		button.setEnabled(True)

	def button_toggle(self, button, func_set, func_get):
		button.setEnabled(False)
		QtGui.QApplication.processEvents()
		try:
			state = func_get()
			if state != button.isChecked():
				func_set(not state)
		except:
			traceback.print_exc()
		button.setEnabled(True)

	def getOpenFileName(self, caption, directory, filters):
		r = QtGui.QFileDialog.getOpenFileName(self, caption, directory, filters)
		if isinstance(r, tuple):
			r = r[0]
		return str(r)

	def getSaveFileName(self, caption, directory, filters):
		r = QtGui.QFileDialog.getSaveFileName(self, caption, directory, filters)
		if isinstance(r, tuple):
			r = r[0]
		return str(r)

	def saveData(self, filename_format=None):

		if filename_format is None:
			filename = self.getSaveFileName("Save PVD", os.getcwd(), "PVD Files (*.pvd)")
			if (filename == ""):
				return
			filename_base, ext = os.path.splitext(filename)
			if ext == "":
				ext = ".pvd"
			filename_format = filename_base + "_%s"
		
		fields = [("mesh", self.problem.mesh), ("rho", self.problem.rho), ("dF", self.problem.dF)]
		if isinstance(self.problem.material_model, TransverseIsotropicMaterialModel):
			fields.append(("p", self.problem.eikonal_p))

		for i, lc in enumerate(self.problem.loadcases):
			fields.append(("u%d" % i, lc.u))
		
		for name, field in fields:
			for ext in ["pvd", "xml"]:
				with File((filename_format % name) + "." + ext) as f:
					f << field


	def saveDensity(self, filename=None):

		if filename is None:
			filename = self.getSaveFileName("Save XML", os.getcwd(), "XML Files (*.xml)")
			if (filename == ""):
				return
			filename_base, ext = os.path.splitext(filename)
			if ext == "":
				ext = ".xml"
			filename_format = filename_base + "_%s"
		
		print filename_format

		with File((filename_format % "mesh") + ".xml") as f:
			f << self.problem.mesh

		with File(filename_base + ".xml") as f:
			f << self.problem.rho

	def loadDensity(self, filename=None):

		if filename is None:
			filename = self.getOpenFileName("Load XML", os.getcwd(), "XML Files (*.xml)")
			if (filename == ""):
				return

		print filename

		self.problem.setDensityFilename(filename)
		self.reset()

	def list_timings(self):
		list_timings(False, [TimingType_wall, TimingType_user, TimingType_system])

	def clear_steps(self):
		self.step_data = []
		self.add_step()

	def show_mesh(self, state):
		self.function_plot.show_grid[0] = state
		self.replot()

	def reset(self):
		self.problem.init()
		self.reset2()

	def reset2(self):
		self.algorithm.init()
		self.rho_old = self.problem.rho.copy(deepcopy=True)
		self.clear_steps()
		self.replot()
	
	def add_step(self, force=False):

		app = QtGui.QApplication.instance()
		
		self.problem.computeGradient()

		sd = OptimizationStepData()
		sd.objective = self.problem.computeObjective()
		sd.gradient_norm = (assemble(self.problem.dF**2*dx)**0.5)/self.problem.mesh_V
		sd.projected_gradient_norm = (assemble(self.get_projected_dF()**2*dx)**0.5)/self.problem.mesh_V
		sd.alpha = float(self.problem.alpha)
		sd.alpha_relax = float(self.problem.alpha_relax)
		self.step_data.append(sd)
		print "step", len(self.step_data), "objective=", sd.objective, "dF_norm=", sd.gradient_norm, "PdF_norm=", sd.projected_gradient_norm

		# TODO: this actuall works and drives the projected gradient towards zero, but may reduce convergence speed
		if len(self.step_data) >= 2 and not force:
			#if self.step_data[-1].projected_gradient_norm > self.step_data[-2].projected_gradient_norm or \
			if self.step_data[-1].objective > self.step_data[-2].objective:
				self.problem.alpha_relax.assign(float(self.problem.alpha_relax)*float(app.parameters["tau"]))
				if app.parameters["dismiss"]:
					self.problem.rho.assign(self.rho_old)
					self.problem.computeGradient()
					del self.step_data[-1]
					print "step dismissed"
			
		self.rho_old.assign(self.problem.rho)
		
	def step(self):
		self.algorithm.step()
		self.add_step()
		self.replot()

	def fixed_step(self):
		
		# normalize gradient
		#dF_norm_2 = assemble(self.problem.dF**2*dx)
		#dF_norm_inf = np.max(np.abs(self.problem.dF.vector().array()))
		#self.problem.dF.vector()[:] /= dF_norm_2

		# perform step
		alpha = float(self.problem.alpha)
		rho = self.problem.rho.vector()
		rho.axpy(-alpha, self.problem.dF.vector())
		self.problem.projectDensity()

		self.add_step(force=True)
		self.replot()

	def refine(self):
		self.problem.refine(2)
		self.reset2()

	def kadapt(self):
		self.problem.kadapt()
		self.reset2()

	def adapt(self):
		self.problem.adapt()
		self.reset2()

	def readapt(self):
		self.problem.readapt()
		self.reset2()

	def coarsen(self):
		self.problem.refine(0.5)
		self.reset2()

	def replot(self):
		for fp in self.function_plots:
			fp.dirty = True
		self.function_plot.plot()

	def check_adjoint(self):
		self.problem.checkAdjoint()

	def closeEvent(self, event):

		app = QtGui.QApplication.instance()
		app.saveWindowState(self, "main")
		#app.settings.setValue("tab", self.tabWidget.currentIndex())
		app.settings.sync()
		event.accept()

		QtGui.QMainWindow.closeEvent(self, event)

	def savePlotPDF(self, fp=None, filename=None, show_mesh=False):

		if filename is None:
			filename = self.getSaveFileName("Save PDF", os.getcwd(), "PDF Files (*.pdf)")
			if (filename == ""):
				return
		
		if fp is None:
			fp = self.tabWidget.currentWidget()
		else:
			self.tabWidget.setCurrentWidget(fp)

		#fp.show()
		#oldTab = self.tabWidget.currentWidget()
		#app = QtGui.QApplication.instance()
		#app.processEvents()

		if fp.gdim > 1:
			fp.show_grid[0] = show_mesh
			ylim = fp.axes.get_ylim()
			xlim = fp.axes.get_xlim()
		else:
			xlim = ylim = [0, 1]

		fp.fig.set_size_inches(10.0, 10.0*(ylim[1]-ylim[0])/(xlim[1]-xlim[0]))
		fp.plot()
		fp.fig.savefig(filename, format='PDF', transparent=False)

		#self.tabWidget.setCurrentWidget(oldTab)


	def savePlot(self, fp, filename_format, show_mesh=False):
		label = fp.label.replace(' ', '_')
		filename = filename_format % label
		print "saving plots", filename

		self.savePlotPDF(fp, filename + ".pdf", show_mesh)

		if fp.gdim > 1:
			for ext in ["pvd", "xml"]:
				with File(filename + "." + ext) as f:
					f << fp.func[-1]()
		with File(filename + "_mesh.xml") as f:
			f << self.problem.mesh

		#self.tabWidget.setCurrentWidget(oldTab)

	def savePlots(self, filename_format, show_mesh=False):
		for fp in self.function_plots:
			if fp.save:
				self.savePlot(fp, filename_format, show_mesh)

	def autoRun(self):

		app = QtGui.QApplication.instance()
		progress = QtGui.QProgressDialog("Optimization is running...", "Cancel", 0, 0, self)
		#progress.setWindowModality(QtCore.Qt.WindowModal)
		progress.show()

		for p in ["write_interval", "num_refinements", "num_adaptions", "num_steps"]:
			locals()[p] = app.parameters[p]

		result_dir = app.parameters["result_dir"]
		if result_dir:

			try:
				os.makedirs(result_dir)
			except:
				pass
			
			stats_filename = os.path.join(result_dir, "stats.csv")
			stats_fh = open(stats_filename, "wb+")
			stats_fh.write("step\tnum_forward\tnum_adjoint\talpha\talpha_relax\tdofs\thmin\thmax\tobjective\trelative_objective\tgrad_norm\tproj_grad_norm\tsolve_time\n")

			data_filename = os.path.join(result_dir, "fig_%03d_%%s")
			plots_filename = os.path.join(result_dir, "fig_%03d_%%s")

			self.saveData(data_filename % 0)
			self.savePlot(self.function_plots[1], (plots_filename % 0) + "_mesh", show_mesh=True)
		else:
			print "WARNING: no --result_dir parameter specified"
		
		start = time.time()
		solve_time = time.time() - start
		hmin = self.problem.mesh_hmin
		hmax = self.problem.mesh_hmax
		dofs = self.problem.mesh.num_vertices()
		alpha_0 = float(self.problem.alpha)
		alpha_relax_0 = float(self.problem.alpha_relax)
		tol = float(self.problem.tol)

		print "autoRun mesh dofs", self.problem.dF.vector().array().shape
		print "autoRun enter loop"

		for k in range(num_adaptions+1):

			app.processEvents()
			if progress.wasCanceled():
				break

			for i in range(num_steps+1):

				app.processEvents()
				if progress.wasCanceled():
					break

				sd = self.step_data[-1]

				if result_dir:
					stats_fh.write("%d\t%d\t%d\t%g\t%d\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n" % (i, self.problem.num_forward, self.problem.num_adjoint,
						sd.alpha, sd.alpha_relax, dofs, hmin, hmax, sd.objective, sd.objective/self.step_data[0].objective,
						sd.gradient_norm, sd.projected_gradient_norm, solve_time))
					stats_fh.flush()

				if ((i % int((num_steps+1)/(num_refinements+1))) == 0 and i > 0):
					print "###refine"
					self.problem.refine()
					self.problem.computeGradient()
					if result_dir:
						self.saveData(data_filename % i)
						self.savePlot(self.function_plots[1], (plots_filename % i) + "_mesh", show_mesh=True)
					hmin = self.problem.mesh_hmin
					hmax = self.problem.mesh_hmax
					dofs = self.problem.mesh.num_vertices()

				if (i % write_interval == 0 and i > 0):
					if result_dir:
						self.savePlots(plots_filename % i, show_mesh=False)
					#self.replot()

				print "autoRun step"
				start = time.time()
				converged = self.algorithm.step()
				self.add_step()

				# check convergence by
				#if self.step_data[-1].projected_gradient_norm/self.step_data[0].projected_gradient_norm < tol:
				#	converged = True

				if converged:
					print "autoRun converged"
					break

				solve_time = time.time() - start

				QtGui.QApplication.processEvents()

			if k < num_adaptions:
				# adaptively refine
				self.problem.kadapt()
				self.problem.computeGradient()
				hmin = self.problem.mesh_hmin
				hmax = self.problem.mesh_hmax
				dofs = self.problem.mesh.num_vertices()
		
		# restore old relaxation factor
		self.problem.alpha.assign(alpha_0)
		self.problem.alpha_relax.assign(alpha_relax_0)

		progress.close()
		print "autoRun exit loop"

		if result_dir:
			stats_fh.close()
			self.saveData(data_filename % num_steps)
			if (num_steps % write_interval != 0):
				self.savePlots(plots_filename % num_steps, show_mesh=False)

		if app.parameters["exit"]:
			print "autoRun app quit"
			app.quit()
			print "autoRun app exit"
			app.exit()
			print "sys exit"
			sys.exit(0)
			return

		self.replot()
		return
		exec ""  # do not remove, required for locals()[..] =



class TopologyOptimizationMainWindow(QtGui.QMainWindow): 
  
	def __init__(self, parent=None): 
		 
		QtGui.QMainWindow.__init__(self, parent) 

		self.control = TopologyOptimizationControl()

		self.setWindowTitle("Topology Optimization Tool -- %s" % repr(sys.argv))
		self.setCentralWidget(self.control)

		app = QtGui.QApplication.instance()
		app.restoreWindowState(self, "main")
		#self.tabWidget.setCurrentIndex(app.settings.value("tab", 0).toInt()[0])

		self.setVisible(True)

		if app.parameters["run"]:
			self.control.autoRun()

	def closeEvent(self, event):

		app = QtGui.QApplication.instance()
		app.saveWindowState(self, "main")
		#app.settings.setValue("tab", self.tabWidget.currentIndex())
		app.settings.sync()
		event.accept()

		QtGui.QMainWindow.closeEvent(self, event)


def main(argv):

	np.random.seed(6)

	app = TopologyOptimizationApp(argv)

	#app.window.control.problem.checkAdjoint()
	#return

	sys.exit(app.exec_())


if __name__ == '__main__':
	main(sys.argv)

