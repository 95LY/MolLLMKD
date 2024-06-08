# import time
# import os
# import torch
# import torch.nn.functional as F
# from torch.optim import Adam
# from torch_geometric.data import DataLoader
# from metric import compute_cla_metric, compute_reg_metric
# import numpy as np
# from libauc.losses import APLoss_SH, AUCMLoss
# from libauc.optimizers import SOAP_ADAM, PESG
# from libauc.datasets import ImbalanceSampler
#
# from tensorboardX import SummaryWriter
import time
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from metric import compute_cla_metric, compute_reg_metric
import numpy as np
# from libauc.losses import APLoss_SH, AUCMLoss(已注释)
from libauc.losses import APLoss, AUCMLoss
# from libauc.optimizers import SOAP_ADAM, PESG(已注释)
from libauc.optimizers import SOAP, PESG
# from libauc.datasets import ImbalanceSampler(已注释)

from tensorboardX import SummaryWriter
from ppo_net import PPO
from torch.distributions import Categorical
from sklearn.decomposition import PCA

eps = 1e-8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



### This is run function for classification tasks
def run_classification(train_dataset, val_dataset, test_dataset, model_1, model_2, num_tasks, epochs, batch_size, vt_batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay, early_stopping, loss_config, metric, log_dir, save_dir, evaluate, args, pre_train=None):

    model_1 = model_1.to(device)
    model_2 = model_2.to(device)
    if pre_train is not None:
        state_key = torch.load(pre_train)
        filtered = {k:v for k,v in state_key.items() if 'mlp1' not in k}
        model_1.load_state_dict(filtered, False)
        model_1.mlp1.reset_parameters()
        model_2.load_state_dict(filtered, False)
        model_2.mlp1.reset_parameters()
    
    loss_type = loss_config['type']
    if loss_type == 'bce':
        criterion_1 = torch.nn.BCEWithLogitsLoss(reduction='none')
        optimizer_1 = Adam(model_1.parameters(), lr=lr, weight_decay=weight_decay)

        criterion_2 = torch.nn.BCEWithLogitsLoss(reduction='none')
        optimizer_2 = Adam(model_2.parameters(), lr=lr, weight_decay=weight_decay)
    elif loss_type == 'auprc':
        labels = [int(data.y.item()) for data in train_dataset]
        margin, beta = loss_config['margin'], loss_config['beta']
        data_len = len(train_dataset) + len(val_dataset) + len(test_dataset)
        criterion_1 = APLoss_SH(margin=margin, beta=beta, data_len=data_len)
        optimizer_1 = SOAP_ADAM(model_1.parameters(), lr=lr, weight_decay=weight_decay)

        criterion_2 = APLoss_SH(margin=margin, beta=beta, data_len=data_len)
        optimizer_2 = SOAP_ADAM(model_2.parameters(), lr=lr, weight_decay=weight_decay)
    elif loss_type == 'auroc':
        labels = [int(data.y.item()) for data in train_dataset]
        margin, gamma, imratio = loss_config['margin'], loss_config['gamma'], sum(labels) / len(labels)
        criterion_1 = AUCMLoss(margin=margin, imratio=imratio)
        a_1, b_1, alpha_1 = criterion_1.a, criterion_1.b, criterion_1.alpha
        optimizer_1 = PESG(model_1, a=a_1, b=b_1, alpha=alpha_1, imratio=imratio, lr=lr, weight_decay=weight_decay, gamma=gamma, margin=margin)

        criterion_2 = AUCMLoss(margin=margin, imratio=imratio)
        a_2, b_2, alpha_2 = criterion_2.a, criterion_2.b, criterion_2.alpha
        optimizer_2 = PESG(model_2, a=a_2, b=b_2, alpha=alpha_2, imratio=imratio, lr=lr, weight_decay=weight_decay, gamma=gamma, margin=margin)
    else:
        raise ValueError('not supported loss function!')
    

    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, vt_batch_size, shuffle=False)
    
    best_val_metric_1 = 0
    best_val_metric_2 = 0
    val_loss_history_1 = []
    val_loss_history_2 = []
    
    epoch_bvl_1 = 0
    epoch_bvl_2 = 0

    
    writer = None
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_dir = os.path.join(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_dir_1 = os.path.join(save_dir, 'params_1.ckpt')
    save_dir_2 = os.path.join(save_dir, 'params_2.ckpt')

    # agent = PPO(input_dim=4, output_dim=2, device=device).to(device)
    r_base = 0
    for epoch in range(1, epochs + 1):
        ### synchronizing helps to record more accurate running time
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if loss_type == 'auprc':
            sampler = ImbalanceSampler(labels, batch_size, pos_num=1)
            train_loader = DataLoader(train_dataset, batch_size, sampler=sampler)
        else:   
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        
        t_start = time.perf_counter()
        
        total_train_loss_1, total_train_loss_2 = train_classification(model_1, model_2, optimizer_1, optimizer_2, train_loader, num_tasks, loss_type, criterion_1, criterion_2, args, epoch, r_base, device)
        
        val_prc_results_1, val_roc_results_1, val_prc_results_2, val_roc_results_2, total_val_loss_1, total_val_loss_2 = val_classification(model_1, model_2, val_loader, num_tasks, loss_type, criterion_1, criterion_2, device)

        
        ### All loss we consider is per-sample loss
        train_loss_per_smaple_1 = total_train_loss_1/len(train_dataset)
        train_loss_per_smaple_2 = total_train_loss_2 / len(train_dataset)
        val_loss_per_smaple_1 = total_val_loss_1/len(val_dataset)
        val_loss_per_smaple_2 = total_val_loss_2 / len(val_dataset)
        
        
        if writer is not None:
            writer.add_scalar('train_loss_per_sample_1', train_loss_per_smaple_1, epoch)
            writer.add_scalar('val_loss_per_smaple_1', val_loss_per_smaple_1, epoch)
            if metric == "prc":
                writer.add_scalar('Val PRC', np.mean(val_prc_results_1), epoch)
            elif metric == "roc":
                writer.add_scalar('Val ROC', np.mean(val_roc_results_1), epoch)

            writer.add_scalar('train_loss_per_sample_2', train_loss_per_smaple_2, epoch)
            writer.add_scalar('val_loss_per_smaple_2', val_loss_per_smaple_2, epoch)
            if metric == "prc":
                writer.add_scalar('Val PRC', np.mean(val_prc_results_2), epoch)
            elif metric == "roc":
                writer.add_scalar('Val ROC', np.mean(val_roc_results_2), epoch)

        ### One possible way to selection model: do testing when val metric is best
        if metric == "prc":
            if np.mean(val_prc_results_1) > best_val_metric_1:
                epoch_bvl_1 = epoch
                best_val_metric_1 = np.mean(val_prc_results_1)
                torch.save(model_1.state_dict(), save_dir_1)
            if np.mean(val_prc_results_2) > best_val_metric_2:
                epoch_bvl_2 = epoch
                best_val_metric_2 = np.mean(val_prc_results_2)
                torch.save(model_2.state_dict(), save_dir_2)
        elif metric == "roc":
            if np.mean(val_roc_results_1) > best_val_metric_1:
                epoch_bvl_1 = epoch
                best_val_metric_1 = np.mean(val_roc_results_1)
                torch.save(model_1.state_dict(), save_dir_1)
            if np.mean(val_roc_results_2) > best_val_metric_2:
                epoch_bvl_2 = epoch
                best_val_metric_2 = np.mean(val_roc_results_2)
                torch.save(model_2.state_dict(), save_dir_2)
        else:
            print("Metric is not consistent with task type!!!")

        ### One possible way to stop training    
        val_loss_history_1.append(val_loss_per_smaple_1)
        if early_stopping > 0 and epoch > epochs // 2 and epoch > early_stopping:
            tmp = torch.tensor(val_loss_history_1[-(early_stopping + 1):-1])
            if val_loss_per_smaple_1 > tmp.mean().item():
                break

        val_loss_history_2.append(val_loss_per_smaple_2)
        if early_stopping > 0 and epoch > epochs // 2 and epoch > early_stopping:
            tmp = torch.tensor(val_loss_history_2[-(early_stopping + 1):-1])
            if val_loss_per_smaple_2 > tmp.mean().item():
                break
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        

        print('Epoch: {:03d}, Training Loss_1: {:.6f}, Val Loss_1: {:.6f}, Val PRC_1 (avg over multitasks): {:.4f}, Val ROC_1 (avg over multitasks): {:.4f}, Duration: {:.2f}'.format(
            epoch, train_loss_per_smaple_1, val_loss_per_smaple_1, np.mean(val_prc_results_1), np.mean(val_roc_results_1), t_end - t_start))
        print(
            'Epoch: {:03d}, Training Loss_2: {:.6f}, Val Loss_2: {:.6f}, Val PRC_2 (avg over multitasks): {:.4f}, Val ROC_2 (avg over multitasks): {:.4f}, Duration: {:.2f}'.format(
                epoch, train_loss_per_smaple_2, val_loss_per_smaple_2, np.mean(val_prc_results_2), np.mean(val_roc_results_2),
                t_end - t_start))

        if epoch % lr_decay_step_size == 0:
            for param_group_1 in optimizer_1.param_groups:
                param_group_1['lr'] = lr_decay_factor * param_group_1['lr']
            for param_group_2 in optimizer_2.param_groups:
                param_group_2['lr'] = lr_decay_factor * param_group_2['lr']
    if writer is not None:
        writer.close()
        
    print('======================')    
    print('Stop training at epoch:', epoch, '; Best val_1 metric before this epoch is:', best_val_metric_1, '; Best val metric achieves at epoch:', epoch_bvl_1)
    print('Stop training at epoch:', epoch, '; Best val_2 metric before this epoch is:', best_val_metric_2,
          '; Best val metric achieves at epoch:', epoch_bvl_2)
    print('======================') 
    
    if evaluate:
        print('Loading trained model and testing...')
        model_1.load_state_dict(torch.load(save_dir_1))
        test_prc_results_1, test_roc_results_1 = test_classification(model_1, test_loader, num_tasks, device)
        model_2.load_state_dict(torch.load(save_dir_2))
        test_prc_results_2, test_roc_results_2 = test_classification(model_2, test_loader, num_tasks, device)


        print('======================')        
        print('Epoch: {:03d}, Test PRC_1 (avg over multitasks): {:.4f}, Test ROC_1 (avg over multitasks): {:.4f}'.format(epoch_bvl_1, np.mean(test_prc_results_1), np.mean(test_roc_results_1)))
        print('======================')
        print('Test PRC for all tasks:', test_prc_results_1)
        print('Test ROC for all tasks:', test_roc_results_1)
        print('======================')

        print('======================')
        print('Epoch: {:03d}, Test PRC_2 (avg over multitasks): {:.4f}, Test ROC_2 (avg over multitasks): {:.4f}'.format(epoch_bvl_2, np.mean(test_prc_results_2), np.mean(test_roc_results_2)))
        print('======================')
        print('Test PRC for all tasks:', test_prc_results_2)
        print('Test ROC for all tasks:', test_roc_results_2)
        print('======================')


    
def train_classification(model_1, model_2, optimizer_1, optimizer_2, train_loader, num_tasks, loss_type, criterion_1, criterion_2, args, epoch, r_base, device):
    model_1.train()
    model_2.train()

    losses_1 = []
    losses_2 = []
    for batch_data in train_loader:
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        batch_data = batch_data.to(device)
        out_x_1, out_edge_1, out_x_clique_1, out_1 = model_1(batch_data)
        out_x_2, out_edge_2, out_x_clique_2, out_2 = model_2(batch_data)

        # criterion
        kl = nn.KLDivLoss(reduction='none')
        criterion_list_1 = torch.nn.BCEWithLogitsLoss(reduction='none')
        criterion_list_2 = torch.nn.BCEWithLogitsLoss(reduction='none')

        # 特别注意(其余代码需要添加)
        if loss_type == 'bce':
            if len(batch_data.y.shape) != 2:
                batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
            target = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in batch_data.y.cpu()])
            target = target.to(device)
        elif loss_type == 'auprc':
            target = batch_data.y
        elif loss_type == 'auroc':
            target = batch_data.y


        loss_list_1_origin = criterion_list_1(out_1, target)
        loss_list_2_origin = criterion_list_2(out_2, target)

        # agent
        loss_list_1_agent = criterion_list_1(out_1, target).detach()
        loss_list_2_agent = criterion_list_2(out_2, target).detach()


        # 计算节点状态

        # 创建一个维度与out_x_1相同的全零张量
        loss_list_1 = torch.zeros(out_x_1.shape[0], num_tasks, device=loss_list_1_origin.device)
        loss_list_2 = torch.zeros(out_x_2.shape[0], num_tasks, device=loss_list_2_origin.device)

        # 将原始张量的值复制到新的张量中
        loss_list_1[:loss_list_1_origin.shape[0], :] = loss_list_1_origin
        loss_list_2[:loss_list_2_origin.shape[0], :] = loss_list_2_origin

        # 构造节点状态
        node_state = torch.cat((F.softmax(out_x_1, dim=1), loss_list_1, F.softmax(out_x_2, dim=1), loss_list_2),
                               dim=1)  # 节点级状态

        # agent_node = PPO(input_dim=node_state.shape[1], output_dim=2, device=device).to(device)
        agent_node = PPO(input_dim=node_state.shape[1], input_dim_train=node_state.shape[0], output_dim=2, device=device).to(device)

        node_prob = agent_node.pi(node_state)

        m_node = Categorical(node_prob)
        a1_node = m_node.sample()
        a2_node = 1 - a1_node

        # 计算边状态

        # 创建一个维度与out_edge_1相同的全零张量
        loss_list_1 = torch.zeros(out_edge_1.shape[0], num_tasks, device=loss_list_1_origin.device)
        loss_list_2 = torch.zeros(out_edge_2.shape[0], num_tasks, device=loss_list_2_origin.device)

        # 将原始张量的值复制到新的张量中
        loss_list_1[:loss_list_1_origin.shape[0], :] = loss_list_1_origin
        loss_list_2[:loss_list_2_origin.shape[0], :] = loss_list_2_origin

        # 构造边状态
        edge_state = torch.cat((F.softmax(out_edge_1, dim=1), loss_list_1, F.softmax(out_edge_2, dim=1), loss_list_2),
                               dim=1)  # 边级状态

        # agent_edge = PPO(input_dim=edge_state.shape[1], output_dim=2, device=device).to(device)
        agent_edge = PPO(input_dim=edge_state.shape[1], input_dim_train=edge_state.shape[0], output_dim=2, device=device).to(device)

        edge_prob = agent_edge.pi(edge_state)

        m_edge = Categorical(edge_prob)
        a1_edge = m_edge.sample()
        a2_edge = 1 - a1_edge


        # 计算子图状态

        # 创建一个维度与out_x_clique_1相同的全零张量
        loss_list_1 = torch.zeros(out_x_clique_1.shape[0], num_tasks, device=loss_list_1_origin.device)
        loss_list_2 = torch.zeros(out_x_clique_2.shape[0], num_tasks, device=loss_list_2_origin.device)

        # 将原始张量的值复制到新的张量中
        loss_list_1[:loss_list_1_origin.shape[0], :] = loss_list_1_origin
        loss_list_2[:loss_list_2_origin.shape[0], :] = loss_list_2_origin

        # 构造子图状态
        x_clique_state = torch.cat((F.softmax(out_x_clique_1, dim=1), loss_list_1, F.softmax(out_x_clique_2, dim=1), loss_list_2),
                               dim=1)  # 节点级状态

        # agent_x_clique = PPO(input_dim=x_clique_state.shape[1], output_dim=2, device=device).to(device)
        agent_x_clique = PPO(input_dim=x_clique_state.shape[1], input_dim_train=x_clique_state.shape[0], output_dim=2,
                         device=device).to(device)

        x_clique_prob = agent_x_clique.pi(x_clique_state)

        m_x_clique = Categorical(x_clique_prob)
        a1_x_clique = m_x_clique.sample()
        a2_x_clique = 1 - a1_x_clique


        # 计算图状态

        graph_state = torch.cat((F.softmax(out_1, dim=1), loss_list_1_origin, F.softmax(out_2, dim=1), loss_list_2_origin),
                               dim=1)  # 图级状态

        # agent_graph = PPO(input_dim=graph_state.shape[1], output_dim=2, device=device).to(device)
        agent_graph = PPO(input_dim=graph_state.shape[1], input_dim_train=graph_state.shape[0], output_dim=2, device=device).to(device)

        graph_prob = agent_graph.pi(graph_state)

        m_graph = Categorical(graph_prob)
        a1_graph = m_graph.sample()
        a2_graph = 1 - a1_graph


        if loss_type == 'bce':
            if len(batch_data.y.shape) != 2:
                batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
            mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in batch_data.y.cpu()]) # Skip those without targets (in PCBA, MUV, Tox21, ToxCast)
            mask = mask.to(device)
            target = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in batch_data.y.cpu()])
            target = target.to(device)
            
            # optimize model_1
            # 交叉熵损失
            loss_1 = args.k1 * criterion_1(out_1, target) * mask
            # # 节点级蒸馏损失
            loss_1 += args.k2 * torch.matmul(a1_node.float(),
                                        kl(F.log_softmax(out_x_1, dim=1), F.softmax(out_x_2.detach(), dim=1)).sum(
                                            dim=1))
            # 边级蒸馏损失
            loss_1 += args.k3 * torch.matmul(a1_edge.float(),
                                        kl(F.log_softmax(out_edge_1, dim=1), F.softmax(out_edge_2.detach(), dim=1)).sum(
                                            dim=1))
            # 子图级蒸馏损失
            loss_1 += args.k4 * torch.matmul(a1_x_clique.float(),
                                        kl(F.log_softmax(out_x_clique_1, dim=1), F.softmax(out_x_clique_2.detach(), dim=1)).sum(
                                            dim=1))
            # 图级蒸馏损失
            loss_1 += args.k5 * torch.matmul(a1_graph.float(),
                                        kl(F.log_softmax(out_1, dim=1), F.softmax(out_2.detach(), dim=1)).sum(
                                            dim=1))
            loss_1 = loss_1.sum()
            loss_1.backward()

            # optimize model_2
            # 交叉熵损失
            loss_2 = args.k1 * criterion_2(out_2, target) * mask
            # 节点级蒸馏损失
            loss_2 += args.k2 * torch.matmul(a2_node.float(),
                                        kl(F.log_softmax(out_x_2, dim=1), F.softmax(out_x_1.detach(), dim=1)).sum(
                                            dim=1))
            # 边级蒸馏损失
            loss_2 += args.k3 * torch.matmul(a2_edge.float(),
                                        kl(F.log_softmax(out_edge_2, dim=1), F.softmax(out_edge_1.detach(), dim=1)).sum(
                                            dim=1))
            # 子图级蒸馏损失
            loss_2 += args.k4 * torch.matmul(a2_x_clique.float(),
                                        kl(F.log_softmax(out_x_clique_2, dim=1), F.softmax(out_x_clique_1.detach(), dim=1)).sum(
                                            dim=1))
            # 图级蒸馏损失
            loss_2 += args.k5 * torch.matmul(a2_graph.float(),
                                        kl(F.log_softmax(out_2, dim=1), F.softmax(out_1.detach(), dim=1)).sum(
                                            dim=1))
            loss_2 = loss_2.sum()
            loss_2.backward()

            # optimize agent
            loss_list_1_agent = loss_list_1_agent.unsqueeze(dim=1)
            loss_list_2_agent = loss_list_2_agent.unsqueeze(dim=1)

            if epoch != 0:
                r = -loss_list_1_agent - loss_list_2_agent - r_base
            else:
                r = -loss_list_1_agent - loss_list_2_agent
            r_base = r

            # 节点
            agent_node.put_data((
                node_state.detach(), a1_node.detach(), node_prob.detach(), r.detach()))
            agent_node.train_net()
            # 边
            agent_edge.put_data((
                edge_state.detach(), a1_edge.detach(), edge_prob.detach(), r.detach()))
            agent_edge.train_net()
            # 子图
            agent_x_clique.put_data((
                x_clique_state.detach(), a1_x_clique.detach(), x_clique_prob.detach(), r.detach()))
            agent_x_clique.train_net()
            # 图
            agent_graph.put_data((
                graph_state.detach(), a1_graph.detach(), graph_prob.detach(), r.detach()))
            agent_graph.train_net()


        elif loss_type == 'auprc':
            target = batch_data.y
            # optimize model_1
            predScore_1 = torch.nn.Sigmoid()(out_1)
            # 交叉熵损失
            loss_1 = args.k1 * criterion_1(predScore_1, target, index_s=batch_data.idx.long())
            # 节点级蒸馏损失
            loss_1 += args.k2 * torch.matmul(a1.float(),
                                        kl(F.log_softmax(out_x_1, dim=1), F.softmax(out_x_2.detach(), dim=1)).sum(
                                            dim=1))
            # 边级蒸馏损失
            loss_1 += args.k3 * torch.matmul(a1.float(),
                                        kl(F.log_softmax(out_edge_1, dim=1), F.softmax(out_edge_2.detach(), dim=1)).sum(
                                            dim=1))
            # 子图级蒸馏损失
            loss_1 += args.k4 * torch.matmul(a1.float(),
                                        kl(F.log_softmax(out_x_clique_1, dim=1),
                                           F.softmax(out_x_clique_2.detach(), dim=1)).sum(
                                            dim=1))
            # 图级蒸馏损失
            loss_1 += args.k5 * torch.matmul(a1.float(),
                                        kl(F.log_softmax(out_1, dim=1), F.softmax(out_2.detach(), dim=1)).sum(
                                            dim=1))
            loss_1 = loss_1.sum()
            loss_1.backward()

            # optimize model_2
            predScore_2 = torch.nn.Sigmoid()(out_2)
            # 交叉熵损失
            loss_2 = args.k1 * criterion_2(predScore_2, target, index_s=batch_data.idx.long())
            # 节点级蒸馏损失
            loss_2 += args.k2 * torch.matmul(a2.float(),
                                        kl(F.log_softmax(out_x_2, dim=1), F.softmax(out_x_1.detach(), dim=1)).sum(
                                            dim=1))
            # 边级蒸馏损失
            loss_2 += args.k3 * torch.matmul(a2.float(),
                                        kl(F.log_softmax(out_edge_2, dim=1), F.softmax(out_edge_1.detach(), dim=1)).sum(
                                            dim=1))
            # 子图级蒸馏损失
            loss_2 += args.k4 * torch.matmul(a2.float(),
                                        kl(F.log_softmax(out_x_clique_2, dim=1),
                                           F.softmax(out_x_clique_1.detach(), dim=1)).sum(
                                            dim=1))
            # 图级蒸馏损失
            loss_2 += args.k5 * torch.matmul(a2.float(),
                                        kl(F.log_softmax(out_2, dim=1), F.softmax(out_1.detach(), dim=1)).sum(
                                            dim=1))
            loss_2 = loss_2.sum()
            loss_2.backward()

            # optimize agent
            loss_list_1_agent = loss_list_1_agent.unsqueeze(dim=1)
            loss_list_2_agent = loss_list_2_agent.unsqueeze(dim=1)

            if epoch != 0:
                r = -loss_list_1_agent - loss_list_2_agent - r_base
            else:
                r = -loss_list_1_agent - loss_list_2_agent
            r_base = r

            # 节点
            agent_node.put_data((
                node_state.detach(), a1_node.detach(), node_prob.detach(), r.detach()))
            agent_node.train_net()
            # 边
            agent_edge.put_data((
                edge_state.detach(), a1_edge.detach(), edge_prob.detach(), r.detach()))
            agent_edge.train_net()
            # 子图
            agent_x_clique.put_data((
                x_clique_state.detach(), a1_x_clique.detach(), x_clique_prob.detach(), r.detach()))
            agent_x_clique.train_net()
            # 图
            agent_graph.put_data((
                graph_state.detach(), a1_graph.detach(), graph_prob.detach(), r.detach()))
            agent_graph.train_net()

        elif loss_type == 'auroc':
            target = batch_data.y
            # optimize model_1
            predScore_1 = torch.nn.Sigmoid()(out_1)
            # 交叉熵损失
            loss_1 = args.k1 * criterion_1(predScore_1, target)
            # 节点级蒸馏损失
            loss_1 += args.k2 * torch.matmul(a1.float(),
                                        kl(F.log_softmax(out_x_1, dim=1), F.softmax(out_x_2.detach(), dim=1)).sum(
                                            dim=1))
            # 边级蒸馏损失
            loss_1 += args.k3 * torch.matmul(a1.float(),
                                        kl(F.log_softmax(out_edge_1, dim=1), F.softmax(out_edge_2.detach(), dim=1)).sum(
                                            dim=1))
            # 子图级蒸馏损失
            loss_1 += args.k4 * torch.matmul(a1.float(),
                                        kl(F.log_softmax(out_x_clique_1, dim=1),
                                           F.softmax(out_x_clique_2.detach(), dim=1)).sum(
                                            dim=1))
            # 图级蒸馏损失
            loss_1 += args.k5 * torch.matmul(a1.float(),
                                        kl(F.log_softmax(out_1, dim=1), F.softmax(out_2.detach(), dim=1)).sum(
                                            dim=1))
            loss_1 = loss_1.sum()
            loss_1.backward(retain_graph=True)

            # optimize model_2
            predScore_2 = torch.nn.Sigmoid()(out_2)
            # 交叉熵损失
            loss_2 = args.k1 * criterion_2(predScore_2, target)
            # 节点级蒸馏损失
            loss_2 += args.k2 * torch.matmul(a2.float(),
                                        kl(F.log_softmax(out_x_2, dim=1), F.softmax(out_x_1.detach(), dim=1)).sum(
                                            dim=1))
            # 边级蒸馏损失
            loss_2 += args.k3 * torch.matmul(a2.float(),
                                        kl(F.log_softmax(out_edge_2, dim=1), F.softmax(out_edge_1.detach(), dim=1)).sum(
                                            dim=1))
            # 子图级蒸馏损失
            loss_2 += args.k4 * torch.matmul(a2.float(),
                                        kl(F.log_softmax(out_x_clique_2, dim=1),
                                           F.softmax(out_x_clique_1.detach(), dim=1)).sum(
                                            dim=1))
            # 图级蒸馏损失
            loss_2 += args.k5 * torch.matmul(a2.float(),
                                        kl(F.log_softmax(out_2, dim=1), F.softmax(out_1.detach(), dim=1)).sum(
                                            dim=1))
            loss_2 = loss_2.sum()
            loss_2.backward(retain_graph=True)

            # optimize agent
            loss_list_1_agent = loss_list_1_agent.unsqueeze(dim=1)
            loss_list_2_agent = loss_list_2_agent.unsqueeze(dim=1)

            if epoch != 0:
                r = -loss_list_1_agent - loss_list_2_agent - r_base
            else:
                r = -loss_list_1_agent - loss_list_2_agent
            r_base = r

            # 节点
            agent_node.put_data((
                node_state.detach(), a1_node.detach(), node_prob.detach(), r.detach()))
            agent_node.train_net()
            # 边
            agent_edge.put_data((
                edge_state.detach(), a1_edge.detach(), edge_prob.detach(), r.detach()))
            agent_edge.train_net()
            # 子图
            agent_x_clique.put_data((
                x_clique_state.detach(), a1_x_clique.detach(), x_clique_prob.detach(), r.detach()))
            agent_x_clique.train_net()
            # 图
            agent_graph.put_data((
                graph_state.detach(), a1_graph.detach(), graph_prob.detach(), r.detach()))
            agent_graph.train_net()
        
        optimizer_1.step()
        losses_1.append(loss_1)
        optimizer_2.step()
        losses_2.append(loss_2)
    return sum(losses_1).item(), sum(losses_2).item()



def val_classification(model_1, model_2, val_loader, num_tasks, loss_type, criterion_1, criterion_2, device):
    model_1.eval()
    model_2.eval()

    preds_1 = torch.Tensor([]).to(device)
    preds_2 = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    losses_1 = []
    losses_2 = []
    for batch_data in val_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            _, _, _, out_1 = model_1(batch_data)
            _, _, _, out_2 = model_2(batch_data)
        
        if loss_type == 'bce':
            if len(batch_data.y.shape) != 2:
                batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
            mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in batch_data.y.cpu()])
            mask = mask.to(device)
            target = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in batch_data.y.cpu()])
            target = target.to(device)
            loss_1 = criterion_1(out_1, target) * mask
            loss_1 = loss_1.sum()
            loss_2 = criterion_2(out_2, target) * mask
            loss_2 = loss_2.sum()
        elif loss_type == 'auprc':
            target = batch_data.y
            predScore_1 = torch.nn.Sigmoid()(out_1)
            loss_1 = criterion_1(predScore_1, target, index_s=batch_data.idx.long())
            predScore_2 = torch.nn.Sigmoid()(out_2)
            loss_2 = criterion_2(predScore_2, target, index_s=batch_data.idx.long())
        elif loss_type == 'auroc':
            target = batch_data.y
            predScore_1 = torch.nn.Sigmoid()(out_1)
            loss_1 = criterion_1(predScore_1, target)
            predScore_2 = torch.nn.Sigmoid()(out_2)
            loss_2 = criterion_2(predScore_2, target)

        losses_1.append(loss_1)
        losses_2.append(loss_2)
        pred_1 = torch.sigmoid(out_1) ### prediction real number between (0,1)
        pred_2 = torch.sigmoid(out_2)  ### prediction real number between (0,1)
        preds_1 = torch.cat([preds_1, pred_1], dim=0)
        preds_2 = torch.cat([preds_2, pred_2], dim=0)
        targets = torch.cat([targets, batch_data.y.view(-1, num_tasks)], dim=0)
    
    prc_results_1, roc_results_1 = compute_cla_metric(targets.cpu().detach().numpy(), preds_1.cpu().detach().numpy(),
                                                  num_tasks)
    prc_results_2, roc_results_2 = compute_cla_metric(targets.cpu().detach().numpy(), preds_2.cpu().detach().numpy(),
                                                  num_tasks)
    
    return prc_results_1, roc_results_1, prc_results_2, roc_results_2, sum(losses_1).item(), sum(losses_2).item()



def test_classification(model, test_loader, num_tasks, device):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            _, _, _, out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        pred = torch.sigmoid(out) ### prediction real number between (0,1)
        preds = torch.cat([preds,pred], dim=0)
        targets = torch.cat([targets, batch_data.y], dim=0)
    prc_results, roc_results = compute_cla_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)
    
    return prc_results, roc_results



### This is run function for regression tasks
def run_regression(train_dataset, val_dataset, test_dataset, model, num_tasks, epochs, batch_size, vt_batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay, early_stopping, metric, log_dir, save_dir, evaluate):

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = torch.nn.MSELoss(reduction='none')

    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, vt_batch_size, shuffle=False)
    
    best_val_metric = float('inf')
    val_loss_history = []

    epoch_bvl = 0
    
    writer = None
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_dir = os.path.join(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
    
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, 'params.ckpt')  
    
    for epoch in range(1, epochs + 1):
        ### synchronizing helps to record more accurate running time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True) # Use drop_last to avoid get a 1 sample batch, in which case BatchNorm does not work.

        t_start = time.perf_counter()
        total_train_loss = train_regression(model, optimizer, train_loader, num_tasks, mse_loss, device) 
        val_mae_results, val_rmse_results, total_val_loss = val_regression(model, val_loader, num_tasks, mse_loss, device)
        
        
        ### All loss we consider is per-sample loss
        train_loss_per_smaple = total_train_loss/len(train_dataset)
        val_loss_per_smaple = total_val_loss/len(val_dataset)
        
        if writer is not None:
            writer.add_scalar('train_loss_per_sample', train_loss_per_smaple, epoch)
            writer.add_scalar('val_loss_per_smaple', val_loss_per_smaple, epoch)
            if metric == "mae":
                writer.add_scalar('Val MAE', np.mean(val_mae_results), epoch)
            elif metric == "rmse":
                writer.add_scalar('Val RMSE', np.mean(val_rmse_results), epoch)
        
        ### One possible way to selection model: do testing when val metric is best
        if metric == "mae":
            if np.mean(val_mae_results) < best_val_metric:
                epoch_bvl = epoch
                best_val_metric = np.mean(val_mae_results)

                torch.save(model.state_dict(), save_dir)
        elif metric == "rmse":
            if np.mean(val_rmse_results) < best_val_metric:
                epoch_bvl = epoch
                best_val_metric = np.mean(val_rmse_results)
                torch.save(model.state_dict(), save_dir)
        else:
            print("Metric is not consistent with task type!!!")
            
        ### One possible way to stop training    
        val_loss_history.append(val_loss_per_smaple)
        if early_stopping > 0 and epoch > epochs // 2 and epoch > early_stopping:
            tmp = torch.tensor(val_loss_history[-(early_stopping + 1):-1])
            if val_loss_per_smaple > tmp.mean().item():
                break
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        
        print('Epoch: {:03d}, Training Loss: {:.6f}, Val Loss: {:.6f}, Val MAE (avg over multitasks): {:.4f}, Val RMSE (avg over multitasks): {:.4f}, Duration: {:.2f}'.format(
            epoch, train_loss_per_smaple, val_loss_per_smaple, np.mean(val_mae_results), np.mean(val_rmse_results), t_end - t_start))

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']
    
        
    if writer is not None:
        writer.close() 
    
    print('======================')    
    print('Stop training at epoch:', epoch, '; Best val metric before this epoch is:', best_val_metric, '; Best val metric achieves at epoch:', epoch_bvl)       
    print('======================') 
    
    if evaluate:
        print('Loading trained model and testing...')
        model.load_state_dict(torch.load(save_dir))
        test_mae_results, test_rmse_results = test_regression(model, test_loader, num_tasks, device)

        print('======================')        
        print('Epoch: {:03d}, Test MAE (avg over multitasks): {:.4f}, Test RMSE (avg over multitasks): {:.4f}'.format(epoch_bvl, np.mean(test_mae_results), np.mean(test_rmse_results)))
        print('======================')
        print('Test MAE for all tasks:', test_mae_results)
        print('Test RMSE for all tasks:', test_rmse_results)
        print('======================')
    
    

def train_regression(model, optimizer, train_loader, num_tasks, mse_loss, device):
    model.train()

    losses = []
    for batch_data in train_loader:
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        loss = mse_loss(out, batch_data.y)
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        losses.append(loss)
    return sum(losses).item()
              

    
def val_regression(model, val_loader, num_tasks, mse_loss, device):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    losses = []
    for batch_data in val_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        loss = mse_loss(out, batch_data.y)
        loss = loss.sum()
        losses.append(loss)
        pred = out 
        preds = torch.cat([preds,pred], dim=0)
        targets = torch.cat([targets, batch_data.y], dim=0)
      
    mae_results, rmse_results = compute_reg_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)
    
    return mae_results, rmse_results, sum(losses).item()



def test_regression(model, test_loader, num_tasks, device):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        pred = out
        preds = torch.cat([preds,pred], dim=0)
        targets = torch.cat([targets, batch_data.y], dim=0)

    mae_results, rmse_results = compute_reg_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)

    return mae_results, rmse_results
