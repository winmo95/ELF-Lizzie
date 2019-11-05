/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

#include "elf/logging/IndexedLoggerFactory.h"
#include "elf/utils/utils.h"
#include "../../../elfgames/go/base/common.h"
using json = nlohmann::json;

namespace elf {
namespace ai {
namespace tree_search {

template <typename Action>
struct NodeResponseT {
  std::vector<std::pair<Action, float>> pi;
  float value;
  bool q_flip = false;
};

using NodeId = int;
const NodeId InvalidNodeId = -1;

template <typename S, typename A>
using ForwardFuncT = std::function<bool(const S& s, const A& a, S* next)>;

template <typename S>
using VisitFuncT = std::function<bool(S* s)>;

template <typename S>
using EvalFuncT = std::function<float(const S& s)>;

// TODO" Do we need StateTrait, ActionTrait, ActorTrait (ssengupta@fb)
// The following can be partial templated.
template <typename S, typename A>
struct StateTrait {
 public:
  static std::string to_string(const S&) {
    return "";
  }

  static bool equals(const S& s1, const S& s2) {
    return s1 == s2;
  }

  static bool moves_since(
      const S& /*s*/,
      size_t* /*next_move_number*/,
      std::vector<A>* /*moves*/) {
    // By default it is not provided.
    return false;
  }
};

template <typename Action>
struct ActionTrait {
 public:
  static std::string to_string(const Action& action) {
    return std::to_string(action);
  }

  // We expect static A default_value().
  static Action default_value() {
    return Action();
  }
};

template <typename Actor>
struct ActorTrait {
 public:
  static std::string to_string(const Actor&) {
    return "";
  }
};

struct Score {
  float q;
  float unsigned_q;
  float prior_probability;
  bool first_visit;
};

struct EdgeInfo {
  // From state.
  float prior_probability;
  NodeId child_node;

  float reward;
  float blackeval;
  int num_visits;
  float virtual_loss;
  
  std::shared_ptr<spdlog::logger> logger_;

  EdgeInfo(float probability)
      : prior_probability(probability),
        child_node(InvalidNodeId),
        reward(0),blackeval(0),
        num_visits(0),
   
        virtual_loss(0) {}

  float getQSA() const {
	return reward /num_visits; //reward;
  }

  int get_winrate(Stone player_){
	int games;
	float eval;
	if(num_visits == 0 ){
               games = num_visits +1 ;
	}else{
		  games = num_visits;
	}
	games += num_visits;//blackeval;
	eval = (reward + num_visits/*blackeval*/) / games;
		
	if(player_ == S_WHITE ){
		eval = 1.0f -eval;
	}
		return 10000 * eval;
   }

  void checkValid() const {
    if (virtual_loss != 0) {
      assert(virtual_loss == 0);
    }
  }

  Score getScore(
      bool flip_q_sign,
      int total_parent_visits,
      float unsigned_default_q) const {
    float r = reward;
    if (flip_q_sign) {
      r = -r;
    }

    // Virtual loss.
    // After flipping, r is the win count (-1 for loss, and +1 for win).
    r -= virtual_loss;
    const int num_visits_with_loss = num_visits + virtual_loss;

    Score s;
    s.q =
        (num_visits_with_loss > 0
             ? r / num_visits_with_loss
             : (flip_q_sign ? -unsigned_default_q : unsigned_default_q));
    s.unsigned_q = (num_visits > 0 ? reward / num_visits : unsigned_default_q);
    s.prior_probability =
        prior_probability / (1 + num_visits) * std::sqrt(total_parent_visits);
    s.first_visit = (num_visits_with_loss == 0);

    return s;
  }

  std::string info(bool verbose = false) const {
    std::stringstream ss;

    if (verbose == false) {
      ss << reward << "/" << num_visits << " (" << getQSA()
         << "), Pr: " << prior_probability << ", child node: " << child_node;
    } else {
      ss << "[" << reward << "/" << num_visits << "]["
         << "vl: " << virtual_loss << "][prob:" << prior_probability
         << "][num_visits:" << num_visits << "]";
    }
    return ss.str();
  }
};

template <typename Action>
struct MCTSPolicy {
  std::vector<std::pair<Action, float>> policy;

  std::string info() const {
    std::stringstream ss;
    ss << "Printing out scores for each action." << std::endl;
    for (size_t i = 0; i < policy.size(); i++) {
      const auto& entry = policy[i];
      ss << "A: " << entry.first << ", Score: " << entry.second << std::endl;
    }
    return ss.str();
  }

  void addAction(const Action& action, float score) {
    policy.push_back(std::make_pair(action, score));
  }

  // AlphaGo use t=1 for first 30 moves.
  void normalize(float t = 1) {
    float exp_sum = 0;
    for (auto& entry : policy) {
      float e = std::pow(entry.second, 1.0 / t);
      entry.second = e;
      exp_sum += e;
    }
    for (auto& entry : policy) {
      entry.second /= exp_sum;
    }
  }

  // Sample from the distribution.
  Action sampleAction(std::mt19937* gen) const {
    size_t i = elf_utils::sample_multinomial(policy, gen);
    return policy[i].first;
  }
};

template <typename Action>
struct MCTSResultT {
  enum RankCriterion { MOST_VISITED = 0, PRIOR = 1, UNIFORM_RANDOM };
  Action best_action;
  Action two_action;
  Action thr_action;
  Action for_action;
  Action five_action;
  Action six_action;
  Action sev_action;
  Action ei_action;
  Action ni_action;
  Action ten_action;
 
  float root_value;
  float getval;
  float max_score;
  float two_score;
  float thr_score;
  float for_score;
  float five_score;
  float six_score;
  float sev_score;
  float ei_score;
  float ni_score;
  float ten_score;

  EdgeInfo best_edge_info;
  EdgeInfo two_edge_info;
  EdgeInfo thr_edge_info;
  EdgeInfo for_edge_info;
  EdgeInfo five_edge_info;
  EdgeInfo six_edge_info;
  EdgeInfo sev_edge_info;
  EdgeInfo ei_edge_info;
  EdgeInfo ni_edge_info;
  EdgeInfo ten_edge_info;

  MCTSPolicy<Action> mcts_policy;
  std::vector<std::pair<Action, EdgeInfo>> action_edge_pairs;
  int total_visits;
  RankCriterion action_rank_method;
  
  // TODO: Constructor should set action_rank_methhohd and
  //       action_edges ssengupta@fb.com
  MCTSResultT()
      : best_action(ActionTrait<Action>::default_value()),
        two_action(ActionTrait<Action>::default_value()),
        thr_action(ActionTrait<Action>::default_value()),
        for_action(ActionTrait<Action>::default_value()),
        five_action(ActionTrait<Action>::default_value()), 
	  six_action(ActionTrait<Action>::default_value()),
	  sev_action(ActionTrait<Action>::default_value()),
	  ei_action(ActionTrait<Action>::default_value()),
	  ni_action(ActionTrait<Action>::default_value()),
	  ten_action(ActionTrait<Action>::default_value()),
	  root_value(0.0),
        getval(std::numeric_limits<float>::lowest()),
        max_score(std::numeric_limits<float>::lowest()),
        two_score(std::numeric_limits<float>::lowest()),
        thr_score(std::numeric_limits<float>::lowest()),
        for_score(std::numeric_limits<float>::lowest()),
        five_score(std::numeric_limits<float>::lowest()),
        six_score(std::numeric_limits<float>::lowest()),
        sev_score(std::numeric_limits<float>::lowest()),
        ei_score(std::numeric_limits<float>::lowest()),
        ni_score(std::numeric_limits<float>::lowest()),
        ten_score(std::numeric_limits<float>::lowest()),
        best_edge_info(0),
        two_edge_info(0),
        thr_edge_info(0),
        for_edge_info(0),
        five_edge_info(0),
        six_edge_info(0),
        sev_edge_info(0),
        ei_edge_info(0),
        ni_edge_info(0),
        ten_edge_info(0),
        total_visits(0),
        action_rank_method(MOST_VISITED) {}

  // TODO: This function should be private and called from the constructor
  //       ssengupta@fb.com
  void addActions(const std::unordered_map<Action, EdgeInfo>& action_edges) {
    static std::mt19937 rng(time(NULL));
    int random_idx = 0;

    assert(action_edges.size() > 0);

    if (action_rank_method == UNIFORM_RANDOM) {
      random_idx = rng() % action_edges.size();
    }

    int index = 0;
    for (const std::pair<Action, EdgeInfo>& action_edge : action_edges) {
      // float score = 0;

      float score = (action_rank_method == MOST_VISITED)
          ? action_edge.second.num_visits
          : (action_rank_method == PRIOR) ? action_edge.second.prior_probability
                                          : 1;

      mcts_policy.addAction(action_edge.first, score);
      action_edge_pairs.push_back(action_edge);
      total_visits += action_edge.second.num_visits;

      if (action_rank_method == UNIFORM_RANDOM) {
        // Choose random action
        if (index == random_idx) {
          max_score = score;
          best_action = action_edge.first;
          best_edge_info = action_edge.second;
             }  
      } else {
        // Choose action with max score
        if (score > max_score) {
		  ten_score = ni_score;
		  ten_action = ni_action;
		  ten_edge_info = ni_edge_info;
		  
		  ni_score = ei_score;
		  ni_action = ni_action;
		  ni_edge_info = ei_edge_info;
		  
		  ei_score = sev_score;
		  ei_action = sev_action;
		  ei_edge_info = sev_edge_info;
		  
		  sev_score = six_score;
		  sev_action = six_action;
		  sev_edge_info = six_edge_info;
		  
		  six_score = five_score;
		  six_action = five_action;
		  six_edge_info = five_edge_info;
	
          five_score = for_score;
          five_action = for_action;
          five_edge_info = for_edge_info;
 
          for_score = thr_score;
          for_action = thr_action;
          for_edge_info = thr_edge_info;
 
          thr_score = two_score;
          thr_action = two_action;
          thr_edge_info = two_edge_info;
 
          two_score = max_score;
          two_action = best_action;
          two_edge_info =  best_edge_info;
 
          max_score = score;
          best_action = action_edge.first;
          best_edge_info = action_edge.second;
        }
        if ((score < max_score) & (score > two_score)) {
		  ten_score = ni_score;
		  ten_action = ni_action;
		  ten_edge_info = ni_edge_info;
		  
		  ni_score = ei_score;
		  ni_action = ni_action;
		  ni_edge_info = ei_edge_info;
		  
		  ei_score = sev_score;
		  ei_action = sev_action;
		  ei_edge_info = sev_edge_info;
		  
		  sev_score = six_score;
		  sev_action = six_action;
		  sev_edge_info = six_edge_info;
		  
		  six_score = five_score;
		  six_action = five_action;
		  six_edge_info = five_edge_info;
		  
          five_score = for_score;
          five_action = for_action;
          five_edge_info = for_edge_info;
 
          for_score = thr_score;
          for_action = thr_action;
          for_edge_info = thr_edge_info;
 
          thr_score = two_score;
          thr_action = two_action;
          thr_edge_info = two_edge_info;
 
          two_score = score;
          two_action = action_edge.first;
          two_edge_info = action_edge.second;
        }
        if ((score < two_score) & (score > thr_score)) {
		  ten_score = ni_score;
		  ten_action = ni_action;
		  ten_edge_info = ni_edge_info;
		  
		  ni_score = ei_score;
		  ni_action = ni_action;
		  ni_edge_info = ei_edge_info;
		  
		  ei_score = sev_score;
		  ei_action = sev_action;
		  ei_edge_info = sev_edge_info;
		  
		  sev_score = six_score;
		  sev_action = six_action;
		  sev_edge_info = six_edge_info;
		  
		  six_score = five_score;
		  six_action = five_action;
		  six_edge_info = five_edge_info;
		  
          five_score = for_score;
          five_action = for_action;
          five_edge_info = for_edge_info;
 
          for_score = thr_score;
          for_action = thr_action;
          for_edge_info = thr_edge_info;
 
          thr_score = score;
          thr_action = action_edge.first;
          thr_edge_info = action_edge.second;

        }
        if ((score < thr_score) & (score > for_score)) {
		  ten_score = ni_score;
		  ten_action = ni_action;
		  ten_edge_info = ni_edge_info;

		  ni_score = ei_score;
		  ni_action = ni_action;
		  ni_edge_info = ei_edge_info;  

		  ei_score = sev_score;
		  ei_action = sev_action;
		  ei_edge_info = sev_edge_info;

		  sev_score = six_score;
		  sev_action = six_action;
		  sev_edge_info = six_edge_info;
		  
		  six_score = five_score;
		  six_action = five_action;
		  six_edge_info = five_edge_info;
		  
          five_score = for_score;
          five_action = for_action;
          five_edge_info = for_edge_info;

          for_score = score;
          for_action = action_edge.first;
          for_edge_info = action_edge.second;
        }
        if ((score < for_score) & (score > five_score)) {
		  ten_score = ni_score;
		  ten_action = ni_action;
		  ten_edge_info = ni_edge_info;
		
		  ni_score = ei_score;
		  ni_action = ni_action;
		  ni_edge_info = ei_edge_info;

		  ei_score = sev_score;
		  ei_action = sev_action;
		  ei_edge_info = sev_edge_info;
		  
		  sev_score = six_score;
		  sev_action = six_action;
		  sev_edge_info = six_edge_info;
		  
		  six_score = five_score;
		  six_action = five_action;
		  six_edge_info = five_edge_info;

          five_score = score;
          five_action = action_edge.first;
          five_edge_info = action_edge.second;
        }
		if ((score < five_score) & (score > six_score)) {
		  ten_score = ni_score;
		  ten_action = ni_action;
		  ten_edge_info = ni_edge_info;
		  
		  ni_score = ei_score;
		  ni_action = ni_action;
		  ni_edge_info = ei_edge_info;
		  
		  ei_score = sev_score;
		  ei_action = sev_action;
		  ei_edge_info = sev_edge_info;
		  
		  sev_score = six_score;
		  sev_action = six_action;
		  sev_edge_info = six_edge_info;
		  
		  six_score = score;
		  six_action = action_edge.first;
		  six_edge_info = action_edge.second;
        }
		if ((score < six_score) & (score > sev_score)) {
		  ten_score = ni_score;
		  ten_action = ni_action;
		  ten_edge_info = ni_edge_info;
		  
		  ni_score = ei_score;
		  ni_action = ni_action;
		  ni_edge_info = ei_edge_info;
		  
		  ei_score = sev_score;
		  ei_action = sev_action;
		  ei_edge_info = sev_edge_info;
		  
		  sev_score = score;
		  sev_action = action_edge.first;
		  sev_edge_info = action_edge.second;
        }
		if ((score < sev_score) & (score > ei_score)) {
		  ten_score = ni_score;
		  ten_action = ni_action;
		  ten_edge_info = ni_edge_info;
		  
		  ni_score = ei_score;
		  ni_action = ni_action;
		  ni_edge_info = ei_edge_info;

		  ei_score = score;
		  ei_action = action_edge.first;
		  ei_edge_info = action_edge.second;
        }
		if ((score < ei_score) & (score > ni_score)) {
		  ten_score = ni_score;
		  ten_action = ni_action;
		  ten_edge_info = ni_edge_info;

		  ni_score = score;
		  ni_action = action_edge.first;
		  ni_edge_info = action_edge.second;

        }
		if ((score < ni_score) & (score > ten_score)) {
		  ten_score = score;
		  ten_action = action_edge.first;
		  ten_edge_info = action_edge.second;

        }
      }
	    index++;
    }
  }

#if 0
  bool feed(float score, const std::pair<Action, EdgeInfo>& action_edge) {
    mcts_policy.addAction(action_edge.first, score);
    action_edge_pairs.push_back(action_edge);
    total_visits += action_edge.second.num_visits;

    if (score > max_score) {
      max_score = score;
      best_action = action_edge.first;
      best_edge_info = action_edge.second;
      return true;
    }
    return false;
  }
#endif

  std::pair<int, EdgeInfo> getRank(const Action& action, RankCriterion rc)
      const {
    // [TODO] not efficient if you want to get rank for multiple actions.
    // TODO: This assignment is a bit odd (ssengupta@fb.com)
    auto ae_pairs = action_edge_pairs;

    using T = std::pair<Action, EdgeInfo>;
    using Func = std::function<bool(const T&, const T&)>;
    Func cmp = nullptr;
    switch (rc) {
      case MOST_VISITED: {
        cmp = [](const T& p1, const T& p2) {
          return p1.second.num_visits > p2.second.num_visits;
        };
        std::sort(ae_pairs.begin(), ae_pairs.end(), cmp);
      } break;
      case PRIOR: {
        cmp = [](const T& p1, const T& p2) {
          return p1.second.prior_probability > p2.second.prior_probability;
        };
        std::sort(ae_pairs.begin(), ae_pairs.end(), cmp);
      } break;
      case UNIFORM_RANDOM:
        break;
      default:
        break;
    }

    for (int i = 0; i < (int)ae_pairs.size(); ++i) {
      if (ae_pairs[i].first == action) {
        return std::make_pair(i, ae_pairs[i].second);
      }
    }
    return std::make_pair(-1, EdgeInfo(0));
  }

  std::string info() const {
    std::stringstream ss;
    ss << "BestA: " << ActionTrait<Action>::to_string(best_action)
       << ", MaxScore: " << max_score << ", Info: " << best_edge_info.info();
    return ss.str();
  }
};

} // namespace tree_search
} // namespace ai
} // namespace elf
