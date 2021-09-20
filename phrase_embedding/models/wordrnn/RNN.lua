local RNN = {}

function RNN.rnn(input_size, rnn_size, n, dropout, last_layer)
  
  -- there are n+1 inputs (hiddens on each layer and x)
  print(last_layer)
  local inputs = {}
  table.insert(inputs, nn.Identity()(last_layer)) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()(last_layer)) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    
    local prev_h = inputs[L+1]
    if L == 1 then
      input_size_L = input_size
      -- local embedded = embedding(inputs[1])
      x = nn.Tanh()(inputs[1])
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- RNN tick
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})

    table.insert(outputs, next_h)
  end
-- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  print('1. Loop ends.')
  local proj = nn.Linear(rnn_size, input_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)
  print('2. Loop ends.')
  --nn.gModule(inputs, outputs)
  return nn.JoinTable(2)(outputs)
end

return RNN
