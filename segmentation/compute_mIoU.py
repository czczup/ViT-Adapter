import argparse, os, json, torch
from PIL import Image
import numpy as np
from torchmetrics import JaccardIndex
from torchmetrics.functional.classification import multilabel_jaccard_index

if __name__ == '__main__':
    #Argomenti: le due cartelle di input sono obbligatorie
    parser = argparse.ArgumentParser(description='Calcolo della mean intersection over union per segmentazione')
    parser.add_argument('ground_truth_folder',type=str,nargs=1,help='Percorso della cartella contenente le label.')
    parser.add_argument('prediction_folder',type=str,nargs=1,help='Percorso della cartella contenente le segmentation mask generate.\nLe prediction devono essere immagini con valori da 0 (background) a num_classes.\
                                                                  \nNel caso di multilabel nella cartella dovranno essere presenti k immagini per ogni ground truth dove k è il numero di classi in quell\'immagine\
                                                                   e ogni immagine deve contenere le binary segmentation mask di una sola classe. Il nome del file deve contenere l\'indece della classe, per esmpio nome_1.png, nome_2.png, l\'indice 0 è già riservato per gestire il background')
    parser.add_argument('mapping_dict',type=str,nargs=1,help='Dizionario per mappare i valori di grigio delle label nelle classi.\nEsempio: {\"0\":0,(per il background) \"valore_grigio_classe_1\":1, ...,  \"valore_grigio_classe_n\":n}.\nSe per una classe ci sono più valori di grigio inserirli entrambi come valori diversi del dizionario.')
    parser.add_argument('--m', dest='multi',action='store_const', const=True, default=False,help='Abilita multilabel (default: non viene considerato il caso multilabel.\nPer YOLO si consiglia di abilitarlo per avere una metrica precisa)')
    parser.add_argument('--s', dest='silent',action='store_const', const=True, default=False,help='Non vengono stampate le metriche per ogni immagine ma solo la media finale')

    args = parser.parse_args()
    gt_folder = args.ground_truth_folder[0]
    pred_folder = args.prediction_folder[0]
    mapping_dict = (args.mapping_dict[0])
    mapping_dict = dict(json.loads(mapping_dict))
    mapping_dict = {int(k):int(v) for k,v in zip(mapping_dict.keys(), mapping_dict.values())}
    n_classes = len(np.unique(list(mapping_dict.values())))
    multi_class = args.multi
    silent = args.silent

    if os.path.isdir(gt_folder) and os.path.isdir(pred_folder):
        pred_imgs = os.listdir(pred_folder)
        gt_imgs = os.listdir(gt_folder)
        if (len(pred_imgs) == len(gt_imgs)) or multi_class:
            pred_imgs.sort()
            gt_imgs.sort()
            stessi_nomi = True
            for name in gt_imgs:
                if not multi_class:
                    remove_char = 4
                else:
                    remove_char = 6
                if name[:-4] not in [img[:-remove_char] for img in pred_imgs]: #Nota: [:-n] serve per togliere gli ultimi n caratteri da una stringa
                    stessi_nomi = False
                    break
            if stessi_nomi:
                scores = []
                if not multi_class:
                    for i, name in enumerate(gt_imgs):
                        current_pred = Image.open(f'{pred_folder}/{pred_imgs[i]}').convert('L')
                        current_gt = Image.open(f'{gt_folder}/{name}').convert('L')
                        current_gt = np.array(current_gt).astype(np.int_)
                        current_gt = np.array(list(map(lambda x: list(map(mapping_dict.get, x)),current_gt)))
                        
                        jaccard = JaccardIndex(task='multiclass', num_classes=n_classes)
                        jac = jaccard(torch.from_numpy(np.array(current_pred).astype(np.uint8)), torch.from_numpy(current_gt.astype(np.uint8))).item()
                        scores.append(jac)
                        if not silent:
                            print()
                            print(f'image {name} IoU: {jac}')
                            print()
                            print('*-'*36)
                else:
                     for i, name in enumerate(gt_imgs):
                        predictions = ([n for n in pred_imgs if n[:-remove_char] == name[:-4] ])
                        predictions.sort()        
                        predictions = [Image.open(f'{pred_folder}/{n}').convert('L') for n in predictions]
                        
                        n_classes_img = len(predictions)
                        predictions = [np.array(p) for p in predictions]
                        mask = np.array(Image.open(f'{gt_folder}/{name}').convert('L'))
                        mask = np.array(mask).astype(np.int_)
                        mask = np.array(list(map(lambda x: list(map(mapping_dict.get, x)),mask)))
                        new_masks = np.zeros((mask.shape[0],mask.shape[1],n_classes_img), np.uint8)
                        for cls in list(zip(range(n_classes_img),list(np.unique(mask))[1:])):
                            class_index,class_id = cls
                            new_masks[:,:,class_index][mask == class_id] = 1
                        jac = multilabel_jaccard_index(torch.from_numpy(np.transpose(np.array(predictions),(1,2,0)).reshape(-1,n_classes_img)), torch.from_numpy(new_masks).reshape(-1,n_classes_img), num_labels = n_classes_img)
                        scores.append(jac)
                        if not silent:
                            print()
                            print(f'image {name} IoU: {jac}')
                            print()
                            print('*-'*12)
                        
                    
                print(f'\nmIoU:{np.mean(scores)}\n')


            else:
                #Se non c'è corrispondenza tra i nomi
                print('Errore: ogni label deve avere lo stesso nome della corrispondente prediction per evitare confronti errati')
        else:
            #Se le due cartelle contengono un numero diverso di file
            print('Errore: Nelle due cartelle c\'è un numero di file diverso')

    else:
        #Se una delle due cartelle non esiste
        if not os.path.isdir(gt_folder):
            print('Errore: La cartella di ground truth non esiste')
        else:
            print('Errore: La cartella di prediction non esiste')