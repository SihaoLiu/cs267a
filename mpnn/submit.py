import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from common import *
from model   import *
from dataset import *
from data import *


def run_submit():

    out_dir = \
        '../final'

    initial_checkpoint = \
        '../result/checkpoint/00147500_model.pth'

    csv_file = out_dir +'/submit/submit-%s-larger.csv'%(initial_checkpoint.split('/')[-1][:-4])


    ## setup  -----------------------------------------------------------------------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/submit', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.submit.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 60 #*2 #280*2 #256*4 #128 #256 #512  #16 #32

    test_dataset = ChampsDataset(
                mode ='test',
                csv  ='test_submit',
                split=None,
                augment=None,
    )
    test_loader  = DataLoader(
                test_dataset,
                sampler     = SequentialSampler(test_dataset),
                batch_size  = batch_size,
                drop_last   = False,
                num_workers = 0,
                pin_memory  = True,
                collate_fn  = null_collate
    )

    log.write('batch_size = %d\n'%(batch_size))
    log.write('test_dataset : \n%s\n'%(test_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(node_dim=NODE_DIM,edge_dim=EDGE_DIM, num_target=NUM_TARGET).cuda()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))


    log.write('%s\n'%(type(net)))
    log.write('\n')


    ## start testing here! ##############################################
    test_num = 0
    test_predict = []
    test_coupling_type  = []
    test_coupling_value = []
    test_id = []

    test_loss = 0

    start = timer()
    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(test_loader):

        net.eval()
        with torch.no_grad():
            node = node.cuda()
            edge = edge.cuda()
            edge_index = edge_index.cuda()
            node_index = node_index.cuda()
            coupling_index = coupling_index.cuda()
            coupling_value = coupling_value.cuda()

            predict = net(node, edge, edge_index, node_index, coupling_index)
            loss = criterion(predict, coupling_value)

        #---
        batch_size = len(infor)
        extracted_id = [infor[b][2] for b in range(batch_size)]
        list_extracted_id = list(np.concatenate(extracted_id))
        test_id.extend(list_extracted_id)

        test_predict.append(predict.data.cpu().numpy())
        test_coupling_type.append(coupling_index[:,2].data.cpu().numpy())
        test_coupling_value.append(coupling_value.data.cpu().numpy())

        test_loss += loss.item()*batch_size
        test_num += batch_size


        print('\r %8d/%8d     %0.2f  %s'%(
            test_num, len(test_dataset),test_num/len(test_dataset),
              time_to_str(timer()-start,'min')),end='',flush=True)


        pass  #-- end of one data loader --
    assert(test_num == len(test_dataset))
    print('\n')

    id  = test_id
    predict  = np.concatenate(test_predict)
    if test_dataset.mode == 'test':
        df = pd.DataFrame(list(zip(id, predict)), columns =['id', 'scalar_coupling_constant'])
        print('head of incorrected result:')
        print(df.head())
        # Merge with value that already exists in train set
        data_dir = '../input/champs-scalar-coupling/'
        train_df = pd.read_csv(data_dir + 'train.csv')
        test_df = pd.read_csv(data_dir + 'test_submit.csv')
        # Merge with Test Set first
        submit_test_df = pd.merge(df, test_df, on = ['id'])
        # Merge with Train Set
        submit_test_train_df = pd.merge(submit_test_df, train_df, how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])
	# Count number of filled from training set
        num_filled = submit_test_train_df['scalar_coupling_constant_y'].count()
        test_size = submit_test_train_df['scalar_coupling_constant_y'].size
        print(f'{num_filled}/{test_size}, {num_filled / test_size * 100}% are corrected from Traiining set')
        # Update from Training Set
        submit_test_train_df['scalar_coupling_constant'] = submit_test_train_df['scalar_coupling_constant_y'].fillna(submit_test_train_df['scalar_coupling_constant_x'])
        # Drop useless column
        final = submit_test_train_df.drop(['scalar_coupling_constant_x', 'scalar_coupling_constant_y', 'molecule_name', 'atom_index_0', 'atom_index_1', 'type', 'id_y'], axis=1).rename(columns={'id_x': 'id'})
        print('head of final result:')
        print(final.head())
        # Write to file
        final.to_csv(csv_file,index=False)

        log.write('id        = %d\n'%len(id))
        log.write('predict   = %d\n'%len(predict))
        log.write('csv_file  = %s\n'%csv_file)

    #-------------------------------------------------------------
    # for debug
    if test_dataset.mode == 'train':
        test_loss = test_loss/test_num

        coupling_value = np.concatenate(test_coupling_value)
        coupling_type  = np.concatenate(test_coupling_type).astype(np.int32)

        mae, log_mae = compute_kaggle_metric( predict, coupling_value, coupling_type,)

        for t in range(NUM_COUPLING_TYPE):
            log.write('\tcoupling_type = %s\n'%COUPLING_TYPE[t])
            log.write('\tmae     =  %f\n'%mae[t])
            log.write('\tlog_mae = %+f\n'%log_mae[t])
            log.write('\n')
        log.write('\n')


        log.write('-- final -------------\n')
        log.write('\ttest_loss = %+f\n'%test_loss)
        log.write('\tmae       =  %f\n'%np.mean(mae))
        log.write('\tlog_mae   = %+f\n'%np.mean(log_mae))
        log.write('\n')


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_submit()

    print('\nsucess!')
