import copy
import torch

from collections import namedtuple
from functools import partial
from typing import Dict, List, Tuple
from itertools import chain

from ...task import SimulatedTask
from ....types import TaskState, TaskType, Device, Architecture
from .globals import *
from .env import *


class ParlaRLNormalizedEnvironment2(ParlaRLBaseEnvironment):
  # [Features]
  # 1. dependency state per device (3 * 4, GCN)
  # 2. num. of visible dependents (1, GCN)
  # 3. device per-state load (3 * 4 = 12, FCN)
  # 4. task type
  gcn_indim = 14
  fcn_indim = 12
  outdim = 4
  device_feature_dim = 3

  def __init__(self):
      pass

  def create_gcn_task_workload_state(
      self, node_id_offset: int, target_task: SimulatedTask,
      devices: List, taskmap: Dict) -> Tuple[torch.tensor, torch.tensor]:
      """
      Create a state that shows task workload states.
      This function creates states not only of the current target task, but also
      its adjacent (possibly k-hops in the future) tasks, and its edge list.
      This state will be an input of the GCN layer.
      """
      lst_node_features = []
      lst_src_edge_index = []
      lst_dst_edge_index = []
      # Create a state of the current task, and append it to the features list.
      lst_node_features.append(
          self.create_task_workload_state(target_task, devices, taskmap))
      # This function temporarily assigns an index to each task.
      # This should match the index on the node feature list and the edge list.
      node_id_offset += 1
      for dependency_id in target_task.dependencies:
          dependency = taskmap[dependency_id]
          # Add a dependency to the edge list
          lst_src_edge_index.append(node_id_offset)
          # 0th task is the target task
          lst_dst_edge_index.append(0)
          lst_node_features.append(
              self.create_task_workload_state(dependency, devices, taskmap))
          node_id_offset += 1
      for dependent_id in target_task.dependents:
          dependent = taskmap[dependent_id]
          # 0th task is the target task
          lst_src_edge_index.append(0)
          # Add a dependent to the edge list
          lst_dst_edge_index.append(node_id_offset)
          lst_node_features.append(
              self.create_task_workload_state(dependent, devices, taskmap))
          node_id_offset += 1
      edge_index = torch.Tensor([lst_src_edge_index, lst_dst_edge_index])
      edge_index = edge_index.to(torch.int64)
      node_features = torch.cat(lst_node_features)
      # Src/dst lists
      assert len(edge_index) == 2
      assert len(node_features) == node_id_offset
      return edge_index, node_features

  def create_task_workload_state(
      self, target_task: SimulatedTask, devices: List,
      taskmap: Dict) -> torch.tensor:
      # TODO(hc): need to be normalized
      # The number of dependencies per-state (dim: 4): MAPPED ~ LAUNCHED
      # Then, # of the dependencies per-devices
      # (dim: # of devices): 0 ~ # of devices
      current_gcn_state = torch.zeros(self.gcn_indim)
      per_device_offset = TaskState.LAUNCHED - TaskState.MAPPED + 1
      for dependency_id in target_task.dependencies:
          dependency = taskmap[dependency_id]
          dependency_state_offset = dependency.state - TaskState.MAPPED
          for dev in dependency.assigned_devices:
              this_device_offset = dev.device_id * per_device_offset
              this_dependency_offset = this_device_offset + dependency_state_offset
              current_gcn_state[this_dependency_offset] = \
                  current_gcn_state[this_dependency_offset] + 1
              print("dependency:", this_dependency_offset, " <- ", current_gcn_state[this_dependency_offset])
      total_num_dependencies = len(target_task.dependencies)
      for device in devices:
          if device.architecture == Architecture.CPU:
              continue
          if total_num_dependencies == 0:
              continue
          this_device_offset = per_device_offset * device.device_id
          for si in range(0, per_device_offset):
              current_gcn_state[this_device_offset + si] = \
                  current_gcn_state[this_device_offset + si] / total_num_dependencies
              print(this_device_offset + si , " is normalized to ", current_gcn_state[this_device_offset + si])
      dependent_feature_offset = per_device_offset * (len(devices) - 1)
      task_type_feature_offset = dependent_feature_offset + 1
      # The number of the dependent tasks
      current_gcn_state[dependent_feature_offset] = len(target_task.dependents) / 100
      if "POTRF" in str(target_task.name):
          current_gcn_state[task_type_feature_offset] = 0.1
      elif "SOLVE" in str(target_task.name):
          current_gcn_state[task_type_feature_offset] = 0.2
      elif "SYRK" in str(target_task.name):
          current_gcn_state[task_type_feature_offset] = 0.3
      elif "GEMM" in str(target_task.name):
          current_gcn_state[task_type_feature_offset] = 0.4
      print(" dependents: ", dependent_feature_offset, " = ",
          current_gcn_state[dependent_feature_offset], ", ", len(target_task.dependents))
      print(target_task, " task type: ", task_type_feature_offset, " = ",
          current_gcn_state[task_type_feature_offset], ", ", len(target_task.dependents))
      # print("gcn state:", current_gcn_state)
      return current_gcn_state.unsqueeze(0)

  def create_device_load_state(self, target_task: SimulatedTask, devices: List,
                               reservable_tasks: Dict, launchable_tasks: Dict,
                               launched_tasks: Dict, total_tasks) -> torch.tensor:
      """
      Create a state that shows devices' workload states.
      This state will be an input of the fully-connected layer.
      """
      current_state = torch.zeros(self.fcn_indim)
      # print("******** Create states:", target_task)
      # Per-state load per-device
      idx = 0
      for device in devices:
          if device.architecture == Architecture.CPU:
              continue
          current_state[idx] = len(reservable_tasks[device]) + len(launchable_tasks[device][TaskType.COMPUTE])
          current_state[idx + 1] = len(launchable_tasks[device][TaskType.DATA])
          current_state[idx + 2] = len(launched_tasks[device])
          idx += self.device_feature_dim
      # Normalization
      idx = 0
      if (total_tasks > 0):
          for device in devices:
              if device.architecture == Architecture.CPU:
                  continue
              for i in range(0, self.device_feature_dim):
                  current_state[idx + i] = current_state[idx + i].item() / total_tasks
              idx += self.device_feature_dim
      return current_state

  def create_state(self, target_task: SimulatedTask, devices: List, taskmap: Dict,
                   spawned_tasks: Dict, mappable_tasks: Dict,
                   reservable_tasks: Dict, launchable_tasks: Dict,
                   launched_tasks: Dict):
      """
      Create the current state.
      The state consists of two feature types:
      1) Device load state: How many tasks of each state are mapped to each device
      2) Task workload state: How many dependencies are on each state, and how many
                              dependent tasks this task has?
      """
      total_tasks = 0
      per_device_total_tasks = []
      for device in devices:
          if device.architecture == Architecture.CPU:
              print("CPU id:", device.device_id)
              continue
          print(device, " id :", device.device_id)
          this_device_total_tasks = \
                         len(reservable_tasks[device]) + \
                         len(launchable_tasks[device][TaskType.COMPUTE]) + \
                         len(launchable_tasks[device][TaskType.DATA]) + \
                         len(launched_tasks[device])
          per_device_total_tasks.append(this_device_total_tasks)
          total_tasks += per_device_total_tasks[device.device_id]
          # TODO(hc): this assumes that all tasks are spawned in advnace.
          # so hard to use that.
      print("Total task:", total_tasks)
      current_device_load_state = self.create_device_load_state(
          target_task, devices, reservable_tasks, launchable_tasks,
          launched_tasks, total_tasks)
      #if total_tasks > 0:
      #    total_tasks /= 10
      edge_index, current_workload_features = self.create_gcn_task_workload_state(
          0, target_task, devices, taskmap)
      print("current dev:", current_device_load_state)
      print("current wkl:", current_workload_features)
      return current_device_load_state, edge_index, current_workload_features

  def calculate_reward(self, given_reward):
      return torch.tensor([[given_reward]], dtype=torch.float)

  def finalize_epoch(self, execution_time):
      pass
