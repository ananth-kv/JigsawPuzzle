require 'loadcaffe'
require 'nn'
require 'cudnn'
require 'cunn'

function createModel(nGPU)
   
   local features = nn.Sequential()
   features:add(cudnn.SpatialConvolution(3, 96, 11, 11, 2, 2, 0, 0, 1))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
   features:add(cudnn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
   features:add(cudnn.SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2, 2))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
   features:add(cudnn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
   features:add(cudnn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1, 1))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1, 2))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1, 2))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
   features:add(nn.View(-1):setNumInputDims(3))
   features:add(nn.Linear(2304, 1024))
  
   features:cuda()
   -- create siamese network
   siamese_encoder = nn.ParallelTable()
   siamese_encoder:add(features)

   for i =1,8 do
      siamese_encoder:add(features:clone('weight','bias', 'gradWeight','gradBias'))
   end
   
   siamese_encoder = makeDataParallel(siamese_encoder:cuda(), nGPU) -- defined in util.lua
   
   --torch.save('siamese.t7', siamese_encoder)
   
   -- create classifier
   local classifier = nn.Sequential()
   classifier:add(nn.Linear(9*1024,4096))
   classifier:add(cudnn.ReLU())
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096,100))

   -- combine everything to get the model
   local model = nn.Sequential()
   model:add(nn.SplitTable(2))
   model:add(siamese_encoder)
   model:add(nn.JoinTable(2))
   model:add(classifier)
   
   --print(model.modules[2].modules[1].modules)
   return model

end

--model = createModel()
