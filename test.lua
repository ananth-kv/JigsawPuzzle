--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local batchNumber
local top1_center, loss
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_center = 0
   loss = 0
   for i=1,nTest/opt.batchSize do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = testLoader:get(indexStart, indexEnd)
            return inputs, labels
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_center = top1_center * 100 / nTest
   loss = loss / (nTest/opt.batchSize) -- because loss is calculated per batch
   testLogger:add{
      ['% top1 accuracy (test set) (center crop)'] = top1_center,
      ['avg loss (test set)'] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t ',
                       epoch, timer:time().real, loss, top1_center))

   print('\n')


end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

-- Permutations
local permutations = torch.load('shuffle.dat')	--Ananth

function testBatch(inputsCPU, labelsCPU)
   batchNumber = batchNumber + opt.batchSize

   -- Ananth ---------------------------------------------
   inputsCPU_permuted, labelsCPU_permuted = createPuzzle(inputsCPU)

   -- shift everything to GPU
   inputs:resize(inputsCPU_permuted:size()):copy(inputsCPU_permuted)
   labels:resize(labelsCPU_permuted:size()):copy(labelsCPU_permuted)
   -- Ananth ---------------------------------------------

   outputs = torch.CudaTensor()
   local output = model:forward(inputs)
   local err = criterion:forward(output, labels)

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss = loss + err
   
   local pred = output:float()
   local _, pred_sorted = pred:sort(2, true)
   for i=1,pred:size(1) do
      local g = labels[i][1]
      if pred_sorted[i][1] == g then top1_center = top1_center + 1 end
   end
   if batchNumber % 1024 == 0 then
      print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, nTest))
   end
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
