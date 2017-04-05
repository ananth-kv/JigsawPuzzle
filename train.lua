--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
require 'image'

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     18,   1e-2,   5e-4, },
        { 19,     29,   5e-3,   5e-4  },
        { 30,     43,   1e-3,   0 },
        { 44,     52,   5e-4,   0 },
        { 53,    1e8,   1e-4,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, loss_epoch

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0
   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels = trainLoader:sample(opt.batchSize)
            return inputs, labels
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   model:clearState()
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- Permutations
local permutations = torch.load('shuffle.dat')	--Ananth

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
   -- inputs:resize(inputsCPU:size()):copy(inputsCPU)
   -- labels:resize(labelsCPU:size()):copy(labelsCPU)

   -- Ananth ---------------------------------------------
   local inputsCPU_permuted, labelsCPU_permuted = createPuzzle(inputsCPU)

   -- shift everything to GPU
   inputs:resize(inputsCPU_permuted:size()):copy(inputsCPU_permuted)
   labels:resize(labelsCPU_permuted:size()):copy(labelsCPU_permuted)
   -- Ananth ---------------------------------------------

   local err  = torch.CudaTensor()
   feval = function(x)
      model:zeroGradParameters()
         
      output = model:forward(inputs)
      err = criterion:forward(output, labels)
   
      local gradOutput = criterion:backward(output, labels)
      model:backward(inputs, gradOutput)
      
      return err, gradParameters
   end
   optim.sgd(feval, parameters, optimState)

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err

   -- top-1 error
   local top1 = 0
   do
      local _,prediction_sorted = output:float():sort(2, true) -- descending
      --torch.save('tempLabels.t7', labels:float()) ----
      --torch.save('tempPredicted.t7', prediction_sorted)  ------
      --print("saved............")
      for i=1,opt.batchSize do
	 if prediction_sorted[i][1] == labels[i][1] then
	    top1_epoch = top1_epoch + 1;
	    top1 = top1 + 1
            --print("top1: ", top1, i)
	 end
      end
      top1 = top1 * 100 / opt.batchSize;
   end
   -- Calculate top-1 error, and print information
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, err, top1,
          optimState.learningRate, dataLoadingTime))

   dataTimer:reset()
end

-- Create puzzles Ananth
function createPuzzle(input_images)

   local oH = 225
   local oW = 225
   local sH = 64
   local sW = 64
   local nc = 3		--channels
   local np = 9		-- patches 3X3
   local nim = input_images:size(1)	--Number of images 
   local puzzles = torch.Tensor(nim,np,nc,sW,sH)
   local labels = torch.Tensor(nim,1)
   
   for ii = 1,nim do   
      local im = input_images[ii]
      local iW = im:size(3)
      local iH = im:size(2)

      --local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
      --local w1 = math.ceil(torch.uniform(1e-2, iW-oW))

      local patches = torch.Tensor(np,nc,sW,sH)

      local count = 1
      for w1=0,oW-oW/3,oW/3 do
           for h1=0,oH-oH/3,oH/3 do  
              w2 = w1 + math.ceil(torch.uniform(1e-2, oW/3-sW))
              h2 = h1 + math.ceil(torch.uniform(1e-2, oH/3-sH))
              patches[count] = image.crop(im, w2, h2, w2 + sW, h2 + sH)
              count = count + 1
    	   end
      end
      
      -- Take a random number between 1 and 100	
      -- This rand_index will be the label
      local rand_index = torch.random(1,100)
      local perm = permutations[rand_index]	-- get the permutation

      local puzzle = patches:index(1,perm:long()) 	-- shuffle the images

      -- finally, copy the puzzle and labels [random index]
      puzzles[ii] = puzzle
      labels[ii] = rand_index

   end
   
   return puzzles, labels
end

