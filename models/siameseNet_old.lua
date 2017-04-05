require 'loadcaffe'
require 'nn'
require 'cudnn'
require 'cunn'

function createModel(nGPU)
   
   local prototxt = '/data/Ananth/code/summer/Puzzle/puzzle_torch/deploy_cfn_jps.prototxt'
   local caffeModel = '/data/Ananth/code/summer/Puzzle/puzzle_torch/cfn_jps.caffemodel'

   local raw = loadcaffe.load(prototxt, caffeModel, 'cudnn')

   -- create features
   local features = nn.Sequential() 
   for i =1,19 do
       features:add(raw.modules[i])
   end   
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
