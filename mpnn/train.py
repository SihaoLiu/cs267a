import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from common  import *
from model   import *
from dataset import *


def do_valid(net, valid_loader):

    valid_num = 0
    valid_predict = []
    valid_coupling_type  = []
    valid_coupling_value = []

    valid_loss = 0
    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(valid_loader):

        #if b==5: break
        net.eval()
        node = node.cuda()
        edge = edge.cuda()
        edge_index = edge_index.cuda()
        node_index = node_index.cuda()

        coupling_value = coupling_value.cuda()
        coupling_index = coupling_index.cuda()

        with torch.no_grad():
            predict = net(node, edge, edge_index, node_index, coupling_index)
            loss = criterion(predict, coupling_value)

        #---
        batch_size = len(infor)
        valid_predict.append(predict.data.cpu().numpy())
        valid_coupling_type.append(coupling_index[:,2].data.cpu().numpy())
        valid_coupling_value.append(coupling_value.data.cpu().numpy())

        valid_loss += batch_size*loss.item()
        valid_num  += batch_size

        print('\r %8d /%8d'%(valid_num, len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --
    assert(valid_num == len(valid_loader.dataset))
    #print('')
    valid_loss = valid_loss/valid_num

    #compute
    predict = np.concatenate(valid_predict)
    coupling_value = np.concatenate(valid_coupling_value)
    coupling_type  = np.concatenate(valid_coupling_type).astype(np.int32)
    mae, log_mae   = compute_kaggle_metric( predict, coupling_value, coupling_type,)

    num_target = NUM_COUPLING_TYPE
    for t in range(NUM_COUPLING_TYPE):
        if mae[t] is None:
            mae[t] = 0
            log_mae[t]  = 0
            num_target -= 1

    mae_mean, log_mae_mean = sum(mae)/num_target, sum(log_mae)/num_target
    #list(np.stack([mae, log_mae]).T.reshape(-1))

    valid_loss = log_mae + [valid_loss,mae_mean, log_mae_mean, ]
    return valid_loss


def run_train():

    out_dir = \
        '../result'

    initial_checkpoint = \
        None#'../result/checkpoint/00012500_model.pth'


    schduler = NullScheduler(lr=0.001)

    ## setup  -----------------------------------------------------------------------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/train', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 28 #*2 #280*2 #256*4 #128 #256 #512  #16 #32


    train_dataset = ChampsDataset(
                csv='train',
                mode ='train',
                #split='debug_split_by_mol.1000.npy', #
                split='train_split_by_mol.80012.npy',
                augment=None,
    )
    train_loader  = DataLoader(
                train_dataset,
                #sampler     = SequentialSampler(train_dataset),
                sampler     = RandomSampler(train_dataset),
                batch_size  = batch_size,
                drop_last   = True,
                num_workers = 8,
                pin_memory  = True,
                collate_fn  = null_collate
    )

    valid_dataset = ChampsDataset(
                csv='train',
                mode='train',
                #split='debug_split_by_mol.1000.npy', # #,None
                split='valid_split_by_mol.5000.npy',
                augment=None,
    )
    valid_loader = DataLoader(
                valid_dataset,
                #sampler     = SequentialSampler(valid_dataset),
                sampler     = RandomSampler(valid_dataset),
                batch_size  = batch_size,
                drop_last   = False,
                num_workers = 0,
                pin_memory  = True,
                collate_fn  = null_collate
    )


    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(node_dim=NODE_DIM,edge_dim=EDGE_DIM, num_target=NUM_TARGET).cuda()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    if initial_checkpoint is not None:
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n'%(type(net)))
    log.write('\n')

    # pretrain_file = '/root/share/project/kaggle/2019/champs_scalar/result/backup/00370000_model.pth'
    # load_pretrain(net,pretrain_file)

    ## optimiser ----------------------------------
    # if 0: ##freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False
    #     pass

    #net.set_mode('train',is_freeze_bn=True)
    #-----------------------------------------------

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.9, weight_decay=0.0001)

    iter_accum  = 1
    num_iters   = 3000  *1000
    iter_smooth = 50
    iter_log    = 500
    iter_valid  = 500
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 2500))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']

            optimizer.load_state_dict(checkpoint['optimizer'])
        pass



    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################

    log.write('** start training here! **\n')
    log.write('   batch_size =%d,  iter_accum=%d\n'%(batch_size,iter_accum))
    log.write('                      |--------------- VALID ----------------------------------------------------------------|-- TRAIN/BATCH ---------\n')
    log.write('                      |std %4.1f    %4.1f    %4.1f    %4.1f    %4.1f    %4.1f    %4.1f   %4.1f  |                    |        | \n'%tuple(COUPLING_TYPE_STD))
    log.write('rate     iter   epoch |    1JHC,   2JHC,   3JHC,   1JHN,   2JHN,   3JHN,   2JHH,   3JHH |  loss  mae log_mae | loss   | time          \n')
    log.write('--------------------------------------------------------------------------------------------------------------------------------------\n')
              #0.00100  111.0* 111.0 | 1.0 +1.2, 2.0 +1.2, 3.0 +1.2, 4.0 +1.2, 5.0 +1.2, 6.0 +1.2, 7.0 +1.2, 8.0 +1.2 | 8.01 +1.21  5.620 | 5.620 | 0 hr 04 min
               #    %5.2f     %5.2f     %5.2f     %5.2f     %5.2f     %5.2f     %5.2f     %5.2f

    train_loss   = np.zeros(20,np.float32)
    valid_loss   = np.zeros(20,np.float32)
    batch_loss   = np.zeros(20,np.float32)
    iter = 0
    i    = 0


    start = timer()
    while  iter<num_iters:
        sum_train_loss = np.zeros(20,np.float32)
        sum = 0

        optimizer.zero_grad()
        for node, edge, edge_index, node_index, coupling_value, coupling_index, infor in train_loader:

            #while 1:
                batch_size = len(infor)
                iter  = i + start_iter
                epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch


                # debug-----------------------------
                # if 0:
                #     pass

                #if 0:
                if (iter % iter_valid==0):
                    valid_loss = do_valid(net, valid_loader) #



                if (iter % iter_log==0):
                    print('\r',end='',flush=True)
                    asterisk = '*' if iter in iter_save else ' '
                    log.write('%0.5f  %5.1f%s %5.1f |  %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f | %+5.3f %5.2f %+0.2f | %+5.3f | %s' % (\
                             rate, iter/1000, asterisk, epoch,
                             *valid_loss[:11],
                             train_loss[0],
                             time_to_str((timer() - start),'min'))
                    )
                    log.write('\n')


                #if 0:
                if iter in iter_save:
                    torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
                    torch.save({
                        'optimizer': optimizer.state_dict(),
                        'iter'     : iter,
                        'epoch'    : epoch,
                    }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                    pass




                # learning rate schduler -------------
                lr = schduler(iter)
                if lr<0 : break
                lr = lr * 0.9925 ** (iter / 1000)
                adjust_learning_rate(optimizer, lr)
                rate = get_learning_rate(optimizer)
                
                # one iteration update  -------------
                #net.set_mode('train',is_freeze_bn=True)

                net.train()
                node = node.cuda()
                edge = edge.cuda()
                edge_index = edge_index.cuda()
                node_index = node_index.cuda()
                coupling_value = coupling_value.cuda()
                coupling_index = coupling_index.cuda()


                predict = net(node, edge, edge_index, node_index, coupling_index)
                loss = criterion(predict, coupling_value)

                (loss/iter_accum).backward()
                if (iter % iter_accum)==0:
                    optimizer.step()
                    optimizer.zero_grad()

                # print statistics  ------------
                batch_loss[:1] = [loss.item()]
                sum_train_loss += batch_loss
                sum += 1
                if iter%iter_smooth == 0:
                    train_loss = sum_train_loss/sum
                    sum_train_loss = np.zeros(20,np.float32)
                    sum = 0


                print('\r',end='',flush=True)
                asterisk = ' '
                print('%0.5f  %5.1f%s %5.1f | %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f | %+5.3f %5.2f %+0.2f | %+5.3f | %s' % (\
                             rate, iter/1000, asterisk, epoch,
                             *valid_loss[:11],
                             batch_loss[0],
                             time_to_str((timer() - start),'min')), end='',flush=True)
                i=i+1


        pass  #-- end of one data loader --
    pass #-- end of all iterations --

    log.write('\n')



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()

    print('\nsucess!')
