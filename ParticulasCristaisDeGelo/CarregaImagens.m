clear

ELEV_INIC = 6;
ELEV_FIM = 25;
NUM_ELEV = ELEV_FIM - ELEV_INIC + 1;
NUM_AZIMUTH = 361;

% h5disp('l36.0_d12.0_flat.h5');
% Intensidades=h5read('l36.0_d12.0_flat.h5','/intensity')';
% Elevation=h5read('l36.0_d12.0_flat.h5','/coordinates/elevation')';
% Orientations=h5read('l36.0_d12.0_flat.h5','/orientations')';
treino = 0.7; % - 70% dos arquivos para treino
teste = 0.3; % - 30% dos arquivos para teste
folder = '../../../../datasets/database1/';

% Le todos os arquivos h5 do diretorio
fnames = dir(strcat(folder,'*.h5'));
numfids = length(fnames);
padrao = cell(1,numfids);
for K = 1:numfids
    name = strcat(folder,fnames(K).name);
    padrao{K}.Intensidades = h5read(name,'/intensity')';
    padrao{K}.Orientations = h5read(name,'/orientations')';
    padrao{K}.Size = h5read(name,'/size')';
    padrao{K}.Elevation = h5read(name,'/coordinates/elevation')';
end

idx=find(padrao{1}.Elevation<=ELEV_FIM & padrao{1}.Elevation>=ELEV_INIC);

NUM_ORIENT = 133;

padrao{1}.orient(1).img=zeros(NUM_ELEV,NUM_AZIMUTH);


for k=1:numfids
    for j=1:NUM_ORIENT
        for i=1:NUM_ELEV
            padrao{k}.orient(j).img(i,:) = padrao{k}.Intensidades(j,idx((i-1)*NUM_AZIMUTH+1:(i-1)*NUM_AZIMUTH+NUM_AZIMUTH));
        end
    end
    %     imagesc(padrao.orient(j).img);
    %     drawnow;
    %     F(j)= getframe;
end

M=zeros(numfids*NUM_ORIENT,NUM_ELEV*NUM_AZIMUTH+1);
for k=1:numfids
    for j=1:NUM_ORIENT
        M((k-1)*NUM_ORIENT+j,:)=[reshape(padrao{k}.orient(j).img,1,NUM_ELEV*NUM_AZIMUTH) padrao{k}.Size(j)];
    end
end
csvwrite('DadosIntensidade.csv',M);

[O,z]=sortrows(padrao{1}.Orientations, [2 3 1]);

F(numfids*NUM_ORIENT)=getframe;
for k=1:numfids
    for j=1:NUM_ORIENT
        image(padrao{k}.orient(z(j)).img);
        colormap(gray)
        alpha=O(j,1);
        beta=O(j,2);
        gamma=O(j,3);
        titulo=sprintf('Orientation: Alpha=%f; Beta=%f; Gamma=%f', alpha,beta,gamma);
        title(titulo);
        drawnow;
        F((k-1)*NUM_ORIENT+j)= getframe;
    end
end
pause;
figure;
movie(F);
    

