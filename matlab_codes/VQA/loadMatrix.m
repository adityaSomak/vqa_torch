function [ num_mat ] = loadMatrix( fname, num_blocks )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

fid = fopen(fname, 'rt');
fprintf('Processing file %s\n',fname);
formatString = repmat('%f ', 1, 300);
formatString (end) = [];
num_mat = zeros(0, 300);
%textscan(fid, '%d %d',1);
blocknum = 0;
while ~feof(fid)
  data = textscan(fid, formatString, 100000);
  num_mat = [num_mat; cell2mat(data)];
  fprintf('%d blocks loaded\n',blocknum);
  blocknum = blocknum+1;
  if num_blocks ~= -1 && blocknum >= num_blocks
      break
  end
end
fclose(fid);

end

