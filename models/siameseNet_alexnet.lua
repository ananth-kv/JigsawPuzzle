require 'loadcaffe'
require 'nn'
require 'cudnn'
require 'cunn'

function createModel(nGPU)
   
   local features = nn.Sequential()
   features:add(nn.SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
   features:add(nn.SpatialBatchNormalization(64,1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   features:add(nn.SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
   features:add(nn.SpatialBatchNormalization(192,1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   features:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(384,1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(256,1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(256,1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
   features:add(nn.View(-1):setNumInputDims(3))
   features:add(nn.Linear(256, 1024))
  
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
