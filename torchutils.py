# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon Mar 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko

"""
import random
import numpy as np
import torch
import torch.optim as optim
import torchsummary

#import gpu.cudautils	#we are not using these yet, it has more power and control over cuda

kAutoDetect=False

if kAutoDetect:
	import pycuda.driver as cuda
	
	class aboutCudaDevices():
		def __init__(self):
			""" pycuda.driver """
			cuda.init()
		
		def num_devices(self):
			"""Return number of devices connected."""
			return cuda.Device.count()
		
		def devices(self):
			"""Get info on all devices connected."""
			devices = []
			num = cuda.Device.count()
			for id in range(num):
				name = cuda.Device(id).name()
				memory = cuda.Device(id).total_memory()
				devices.append((memory, name, id))
			return devices

		def preferred(self):
			""" return preferred cuda device - (<memory>, <name>, <id>) """
			#1: sort by memory size	- use better smarts when needed
			devices = sorted(self.devices(), reverse=True)
			return devices

		def mem_info(self):
			"""Get available and total memory of all devices."""
			available, total = cuda.mem_get_info()  #Note: pycuda._driver.LogicError: cuMemGetInfo failed: context is destroyed
			print("Available: %.2f GB\nTotal:     %.2f GB"%(available/1e9, total/1e9))
			
		def attributes(self, device_id=0):
			"""Get attributes of device with device Id = device_id"""
			return cuda.Device(device_id).get_attributes()
		
		def __repr__(self):
			"""Class representation as number of devices connected and about them."""
			num = cuda.Device.count()
			string = ""
			string += ("%d device(s) found:\n"%num)
			for i in range(num):
				string += ( "    %d) %s (Id: %d)\n"%((i+1),cuda.Device(i).name(),i))
				string += ("          Memory: %.2f GB\n"%(cuda.Device(i).total_memory()/1e9))
			return string

	def get_cuda(cudadevice='cuda:0'):
		""" return the best Cuda device """
		#print ('Available cuda devices ', torch.cuda.device_count())
		if cudadevice is None:
			about = aboutCudaDevices()
			devid = about.preferred()[2]
		else:		
			devid = cudadevice
		#print ('Current cuda device ', devid, torch.cuda.get_device_name(devid))
		#device = 'cuda:0'	#most of the time torch choose the right CUDA device
		return torch.device(devid)		#use this device object instead of the device string
else:
	def get_cuda(cudadevice='cuda:0'):
		""" return the best Cuda device """
		devid = cudadevice
		#print ('Current cuda device ', devid, torch.cuda.get_device_name(devid))
		#device = 'cuda:0'	#most of the time torch choose the right CUDA device
		return torch.device(devid)		#use this device object instead of the device string
#if kAutoDetect

def initSeeds():
	random.seed(1)
	torch.manual_seed(1)
	torch.cuda.manual_seed(1)
	np.random.seed(1)

def onceInit(kCUDA=False, cudadevice='cuda:0'):
	print(f"onceInit {cudadevice}")
	if kCUDA and torch.cuda.is_available():
		if cudadevice is None:
			device = get_cuda()
		else:
			device = torch.device(cudadevice)
			torch.cuda.set_device(device)
	else:
		device = 'cpu'

	print(f"torchutils.onceInit device = {device}")
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.enabled = kCUDA

	initSeeds()

	return device

def allocatedGPU():
	# Returns the current GPU memory usage by 
	# tensors in bytes for a given device
	return torch.cuda.memory_allocated()	

def shutdown():
	# Releases all unoccupied cached memory currently held by
	# the caching allocator so that those can be used in other
	# GPU application and visible in nvidia-smi
	torch.cuda.empty_cache()	

#https://pytorch.org/tutorials/beginner/saving_loading_models.html
def save_snapshot(epoch, net, optimizer, running_loss, snapshot_name):
	""" create a snapshot for the model's parameters and optionally the optimizer's state """
	state =	{
#		'epoch':	epoch+1,	#TODO: support saving epoch later - for continue training
		'model_state_dict': net.state_dict(), 
		'loss': running_loss,
	}
	if optimizer:	#this makes the snapshot much bigger
		state.update({'optimizer_state_dict': optimizer.state_dict()})
	torch.save(
		state,
		snapshot_name+'.pth'
#		snapshot_name+str(epoch+1)+'.pth'	#TODO: encode the epoch we stopped at
	)

def load_snapshot(device, net, snapshot_name, optimizer=None):
	""" load a snapshot into 'device' and restore the model_state """
	try:
		checkpoint = torch.load(snapshot_name+'.pth', map_location=device)
		net.load_state_dict(checkpoint['model_state_dict'])
		if optimizer:
			restore_optimizer(optimizer, checkpoint)
	except:
		checkpoint = None	
	return checkpoint

def load1model(device, folder, snapshot_name, model, epoch=None):
	print(f" load model '{snapshot_name}'", end='')
	snapshot = load_snapshot(device, model, snapshot_name=snapshot_name, optimizer=None)
	state_dict = model.state_dict()
	print(f", statedict {len(state_dict)}")
	return snapshot

def restore_optimizer(optimizer, snapshot):
	optim_state = snapshot.get('optimizer_state_dict', None)
	if optim_state:
		optimizer.load_state_dict(optim_state)


def is_cuda_device(device):
	""" predicate for 'device' being a cuda device """
	return 'cuda' in str(device)

def is_cuda(model):
	""" predicate for 'model' begin on a Cuda device """
	return next(model.parameters()).is_cuda

def log_summary(model, ip_shape, logfile):
	successful = False
	try:
		with open(logfile, 'wt') as fout:
			summary, _ = torchsummary.summary_string(model, ip_shape, device = 'cpu')
			fout.write(summary)
			sucessful = True
	except:
		print(f"Error logging :> model : {model.__class__.__name__}, ip_shape : {ip_shape}, logfile : {logfile} ")
	return successful

def choose_optimizer(optimizer, optim_params):
	""" choose the selected optimizer based on its name """    
	if(optimizer == 'adam'):
		optimizer = optim.Adam(**optim_params)
	elif(optimizer == 'adamw'):
		optimizer = optim.AdamW(**optim_params)
	elif(optimizer == 'asgd'):
		optimizer = optim.ASGD(**optim_params)
	elif(optimizer == 'adamax'):
		optimizer = optim.Adamax(**optim_params)
	elif(optimizer == 'adagrad'):
		optimizer = optim.Adagrad(**optim_params)
	elif(optimizer == 'asgd'):
		optimizer = optim.ASGD(**optim_params)
	elif(optimizer == 'rmsprop'):
		optimizer = optim.RMSprop(**optim_params)
	return optimizer

def getBatch(dataset, indices):
	print(type(indices))
	batch = np.sort(indices)

def dumpModelSize(model, details=True):
	total = sum(p.numel() for p in model.parameters())
	if details:
		for name, param in model.named_parameters():
			if param.requires_grad:
				num_params = sum(p.numel() for p in param)
				print(f"name: {name}, num params: {num_params} ({(num_params/total) *100 :.2f}%)")

	print(f"total params: {total}, ", end='')
	print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

def dumpLayers(model):
	for name, layer in model.named_modules():
		if name is not '':	#do not dump the model itself
			print(f" {name}: {layer}")		

def modelName(model):
	modelname = getattr(model, "_modelName", model.__class__.__name__)
	return modelname
	