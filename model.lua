--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'optim'
require 'cudnn'

--[[
   1. Create Model
   2. Create Criterion
   3. Convert model to CUDA
]]--

-- 1. Create Network
-- 1.1 If preloading option is set, preload weights from existing models appropriately

-- Ananth
local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))

         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
end

local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
	 v.bias:zero()
      end
end
-- Ananth


if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = loadDataParallel(opt.retrain, opt.nGPU) -- defined in util.lua
   
else
   --paths.dofile('models/siameseNet.lua')
   paths.dofile('models/siameseNet_alexnet.lua')
   print('=> Creating mode')
   model = createModel(opt.nGPU) -- for the model creation code, check the models/ folder
   --model = torch.load('models/puzzle_actual_alexnet.t7')
   
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end

   -- Initialization
   ConvInit('cudnn.SpatialConvolution')
   --BNInit('cudnn.SpatialBatchNormalization')
   
   if opt.backend == 'cudnn' then
      model = model:cuda()
      cudnn.convert(model, cudnn)
   elseif opt.backend == 'cunn' then
      require 'cunn'
      model = model:cuda()
   elseif opt.backend ~= 'nn' then
      error'Unsupported backend'
   end
end



-- 2. Create Criterion
--criterion = nn.ClassNLLCriterion()
criterion = nn.CrossEntropyCriterion()

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
-- model is converted to CUDA in the init script itself
-- model = model:cuda()
criterion:cuda()

collectgarbage()
