local cuda = pcall(require, 'cutorch') -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available


function load_data(wordMappingFile)
  local test

  print('loading data...')
  assert(opt.data ~= '', 'must provide hdf5 datafile')
  local f = hdf5.open(opt.data, 'r')
  local w2v = f:read('w2v'):all()

  if f:read('test'):dataspaceSize()[1] == 0 then
    opt.has_test = 0
  else
    opt.has_test = 1
    test = f:read('test'):all()
  end
  print('data loaded!')

  idx_to_word = {}
  for line in io.lines(wordMappingFile) do
  	tokens = split(line,"\t");
  	idx_to_word[tokens[1]] = tokens[0]
  end

  return test, w2v, idx_to_word
end

function main()
	local XTest, w2v, idx_to_word = load_data('custom_word_mapping.txt')

	autoencoder = torch.save('phrasemodel5.t7');
	if cuda then
    	XTest = XTest:cuda()
    	autoencoder:cuda()
  	end
  	-- Test
  	print('Testing')
  	x = XTest:narrow(1, 1, 100)
  	local xHat
  	autoencoder:evaluate()
  	xHat = autoencoder:forward(x)
  	-- TODO --
  	[[1. First convert xHat (100 28 60000) to (100 28)
  	2. Get Words from x
  	3. Get Words from xHat]]
  	-- TODO --


main()