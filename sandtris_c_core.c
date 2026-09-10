
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int cleared;
  float flow;
  float bridge;
  int gap;
  int max_height;
  int bumpiness;
  int useful_frontier[5];
  int color_gaps[5];
  bool valid;
} EvalFrontierResult;

#define GRID_WIDTH 85
#define GRID_HEIGHT 150

typedef struct {
  int cleared;
  int gap_active[5];
  int gap_blockage[5];
  int max_height;
  int bumpiness;
  int comp_count[5];
  int max_comp_size[5];
  int exposed_pixels[5];
  bool valid;
} EvalResult;

typedef struct {
  int x, y;
} Point;

static Point stack[GRID_WIDTH * GRID_HEIGHT];
static int stack_ptr = 0;

static inline void push(int x, int y) {
  stack[stack_ptr].x = x;
  stack[stack_ptr].y = y;
  stack_ptr++;
}

static inline Point pop() {
  stack_ptr--;
  return stack[stack_ptr];
}

void step_sand(uint8_t world[GRID_HEIGHT][GRID_WIDTH]) {
  for (int iter = 0; iter < 200; iter++) {
    bool changed = false;

    // Vertical Drop, Gravity
    for (int y = GRID_HEIGHT - 2; y >= 0; y--) {
      for (int x = 0; x < GRID_WIDTH; x++) {
        if (world[y][x] != 0 && world[y + 1][x] == 0) {
          world[y + 1][x] = world[y][x];
          world[y][x] = 0;
          changed = true;
        }
      }
    }

    // Diagonal Slide
    for (int y = GRID_HEIGHT - 2; y >= 0; y--) {
      for (int x = 0; x < GRID_WIDTH; x++) {
        uint8_t val = world[y][x];
        if (val != 0 && world[y + 1][x] != 0) {
          int d1 = ((x + y + iter) % 2 == 0) ? -1 : 1;
          int d2 = -d1;

          if (x + d1 >= 0 && x + d1 < GRID_WIDTH && world[y + 1][x + d1] == 0) {
            world[y + 1][x + d1] = val;
            world[y][x] = 0;
            changed = true;
          } else if (x + d2 >= 0 && x + d2 < GRID_WIDTH &&
                     world[y + 1][x + d2] == 0) {
            world[y + 1][x + d2] = val;
            world[y][x] = 0;
            changed = true;
          }
        }
      }
    }
    if (!changed)
      break;
  }
}

__attribute__((visibility("default"))) EvalResult
evaluate_board(uint8_t world[GRID_HEIGHT][GRID_WIDTH]) {
  EvalResult res;
  memset(&res, 0, sizeof(EvalResult));
  res.valid = true;
  for (int i = 0; i < 5; i++) {
    res.gap_active[i] = 2000;
    res.gap_blockage[i] = 2000;
  }

  int heights[GRID_WIDTH] = {0};

  // Find heights for each column
  for (int x = 0; x < GRID_WIDTH; x++) {
    for (int y = 0; y < GRID_HEIGHT; y++) {
      if (world[y][x] != 0) {
        heights[x] = GRID_HEIGHT - y;
        if (heights[x] > res.max_height)
          res.max_height = heights[x];
        break;
      }
    }
  }

  if (res.max_height == 0)
    return res;

  // Calculate bumpiness
  res.bumpiness = 0;
  for (int x = 0; x < GRID_WIDTH - 1; x++) {
    int diff = heights[x] - heights[x + 1];
    if (diff < 0)
      diff = -diff;
    res.bumpiness += diff;
  }

  // Determine exposed pixels quickly (touching empty space)
  bool exposed[GRID_HEIGHT][GRID_WIDTH] = {0};
  int global_min_x[5] = {GRID_WIDTH, GRID_WIDTH, GRID_WIDTH, GRID_WIDTH,
                         GRID_WIDTH};
  int global_max_x[5] = {-1, -1, -1, -1, -1};
  bool global_left[5] = {false};
  bool global_right[5] = {false};

  for (int y = 0; y < GRID_HEIGHT; y++) {
    for (int x = 0; x < GRID_WIDTH; x++) {
      uint8_t c = world[y][x];
      if (c > 0) {
        bool is_exp = false;
        if (y > 0 && world[y - 1][x] == 0)
          is_exp = true;
        else if (y < GRID_HEIGHT - 1 && world[y + 1][x] == 0)
          is_exp = true;
        else if (x > 0 && world[y][x - 1] == 0)
          is_exp = true;
        else if (x < GRID_WIDTH - 1 && world[y][x + 1] == 0)
          is_exp = true;
        else if (y == 0)
          is_exp = true; // Top edge is always exposed

        if (is_exp) {
          exposed[y][x] = true;
          res.exposed_pixels[c]++;
          if (x < global_min_x[c])
            global_min_x[c] = x;
          if (x > global_max_x[c])
            global_max_x[c] = x;
          if (x == 0)
            global_left[c] = true;
          if (x == GRID_WIDTH - 1)
            global_right[c] = true;
        }
      }
    }
  }

  // Calculate gap_blockage (Global Bounding Box)
  for (int c = 1; c < 5; c++) {
    if (global_max_x[c] >= 0) {
      if (global_left[c] && global_right[c]) {
        res.gap_blockage[c] = 0;
      } else if (global_left[c]) {
        res.gap_blockage[c] = (GRID_WIDTH - 1) - global_max_x[c];
      } else if (global_right[c]) {
        res.gap_blockage[c] = global_min_x[c];
      } else {
        res.gap_blockage[c] =
            1000 + ((global_min_x[c] < (GRID_WIDTH - 1 - global_max_x[c]))
                        ? global_min_x[c]
                        : (GRID_WIDTH - 1 - global_max_x[c]));
      }
    }
  }

  bool visited[GRID_HEIGHT][GRID_WIDTH] = {0};

  // Calculate gap_active (Connected Components)
  for (int y = 0; y < GRID_HEIGHT; y++) {
    for (int x = 0; x < GRID_WIDTH; x++) {
      uint8_t c = world[y][x];
      if (c > 0 && !visited[y][x]) {
        stack_ptr = 0;
        visited[y][x] = true;
        push(x, y);

        res.comp_count[c]++;

        bool left_touch = false;
        bool right_touch = false;
        int exposed_min_x = GRID_WIDTH;
        int exposed_max_x = -1;
        int comp_size = 0;

        while (stack_ptr > 0) {
          Point p = pop();
          comp_size++;

          if (p.x == 0)
            left_touch = true;
          if (p.x == GRID_WIDTH - 1)
            right_touch = true;

          if (exposed[p.y][p.x]) {
            if (p.x < exposed_min_x)
              exposed_min_x = p.x;
            if (p.x > exposed_max_x)
              exposed_max_x = p.x;
          }

          int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
          for (int i = 0; i < 4; i++) {
            int nx = p.x + dirs[i][0];
            int ny = p.y + dirs[i][1];
            if (nx >= 0 && nx < GRID_WIDTH && ny >= 0 && ny < GRID_HEIGHT) {
              if (!visited[ny][nx] && world[ny][nx] == c) {
                visited[ny][nx] = true;
                push(nx, ny);
              }
            }
          }
        }

        if (comp_size > res.max_comp_size[c]) {
          res.max_comp_size[c] = comp_size;
        }

        if (left_touch && right_touch) {
          res.cleared += comp_size;
          res.gap_active[c] = 0;
        } else if (exposed_max_x >= 0) {
          int gap;
          if (left_touch)
            gap = (GRID_WIDTH - 1) - exposed_max_x;
          else if (right_touch)
            gap = exposed_min_x;
          else {
            int gap_L = exposed_min_x;
            int gap_R = (GRID_WIDTH - 1) - exposed_max_x;
            gap = 1000 + ((gap_L < gap_R) ? gap_L : gap_R);
          }

          if (gap < res.gap_active[c]) {
            res.gap_active[c] = gap;
          }
        }
      }
    }
  }

  return res;
}

__attribute__((visibility("default"))) EvalResult
simulate_action(uint8_t original_world[GRID_HEIGHT][GRID_WIDTH], int *grains_x,
                int *grains_y, int num_grains, int color_idx) {
  uint8_t world[GRID_HEIGHT][GRID_WIDTH];
  memcpy(world, original_world, sizeof(world));

  for (int i = 0; i < num_grains; i++) {
    int x = grains_x[i];
    int y = grains_y[i];
    if (x < 0 || x >= GRID_WIDTH || y < 0 || y >= GRID_HEIGHT ||
        world[y][x] != 0) {
      EvalResult res;
      memset(&res, 0, sizeof(res));
      res.valid = false;
      return res;
    }
  }

  bool can_drop = true;
  int dy = 0;
  while (can_drop) {
    for (int i = 0; i < num_grains; i++) {
      int nx = grains_x[i];
      int ny = grains_y[i] + dy + 1;
      if (ny >= GRID_HEIGHT || world[ny][nx] != 0) {
        can_drop = false;
        break;
      }
    }
    if (can_drop)
      dy++;
  }

  for (int i = 0; i < num_grains; i++) {
    int nx = grains_x[i];
    int ny = grains_y[i] + dy;
    world[ny][nx] = color_idx;
  }

  step_sand(world);
  EvalResult res = evaluate_board(world);
  res.valid = true;
  return res;
}

__attribute__((visibility("default"))) void
simulate_all_actions(uint8_t original_world[GRID_HEIGHT][GRID_WIDTH],
                     int num_actions, int *grains_x_flat, int *grains_y_flat,
                     int *num_grains_array, int color_idx,
                     EvalResult *out_results) {
  int grain_idx = 0;
  for (int a = 0; a < num_actions; a++) {
    uint8_t world[GRID_HEIGHT][GRID_WIDTH];
    memcpy(world, original_world, sizeof(world));

    int num_grains = num_grains_array[a];
    int *grains_x = &grains_x_flat[grain_idx];
    int *grains_y = &grains_y_flat[grain_idx];
    grain_idx += num_grains;

    bool valid = true;
    for (int i = 0; i < num_grains; i++) {
      int nx = grains_x[i];
      int ny = grains_y[i];
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT ||
          world[ny][nx] != 0) {
        valid = false;
        break;
      }
    }

    if (!valid) {
      memset(&out_results[a], 0, sizeof(EvalResult));
      out_results[a].valid = false;
      continue;
    }

    bool can_drop = true;
    int dy = 0;
    while (can_drop) {
      for (int i = 0; i < num_grains; i++) {
        int nx = grains_x[i];
        int ny = grains_y[i] + dy + 1;
        if (ny >= GRID_HEIGHT || world[ny][nx] != 0) {
          can_drop = false;
          break;
        }
      }
      if (can_drop)
        dy++;
    }

    for (int i = 0; i < num_grains; i++) {
      int nx = grains_x[i];
      int ny = grains_y[i] + dy;
      world[ny][nx] = color_idx;
    }

    step_sand(world);
    out_results[a] = evaluate_board(world);
    out_results[a].valid = true;
  }
}

__attribute__((visibility("default"))) EvalFrontierResult
evaluate_frontier_board(uint8_t world[GRID_HEIGHT][GRID_WIDTH]) {
  EvalFrontierResult res;
  memset(&res, 0, sizeof(EvalFrontierResult));
  res.valid = true;
  for (int i = 1; i < 5; i++) {
    res.color_gaps[i] = GRID_WIDTH;
  }

  int heights[GRID_WIDTH] = {0};
  for (int x = 0; x < GRID_WIDTH; x++) {
    for (int y = 0; y < GRID_HEIGHT; y++) {
      if (world[y][x] != 0) {
        heights[x] = GRID_HEIGHT - y;
        if (heights[x] > res.max_height)
          res.max_height = heights[x];
        break;
      }
    }
  }

  if (res.max_height == 0)
    return res;

  res.bumpiness = 0;
  for (int x = 0; x < GRID_WIDTH - 1; x++) {
    int diff = heights[x] - heights[x + 1];
    if (diff < 0)
      diff = -diff;
    res.bumpiness += diff;
  }

  // Identify sky reachable (top air) using flood fill from y=0
  bool air_mask[GRID_HEIGHT][GRID_WIDTH] = {0};
  int q_x[GRID_WIDTH * GRID_HEIGHT];
  int q_y[GRID_WIDTH * GRID_HEIGHT];
  int q_head = 0, q_tail = 0;

  for (int x = 0; x < GRID_WIDTH; x++) {
    if (world[0][x] == 0) {
      air_mask[0][x] = true;
      q_x[q_tail] = x;
      q_y[q_tail] = 0;
      q_tail++;
    }
  }

  while (q_head < q_tail) {
    int x = q_x[q_head];
    int y = q_y[q_head];
    q_head++;

    int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    // diagonal is allowed for 8-way adjacent? Scipy structure was cross +
    // corners? Wait, scipy structure=[[0,1,0],[1,1,1],[0,1,0]] is 4-way
    // adjacent!
    for (int i = 0; i < 4; i++) {
      int nx = x + dirs[i][0];
      int ny = y + dirs[i][1];
      if (nx >= 0 && nx < GRID_WIDTH && ny >= 0 && ny < GRID_HEIGHT) {
        if (!air_mask[ny][nx] && world[ny][nx] == 0) {
          air_mask[ny][nx] = true;
          q_x[q_tail] = nx;
          q_y[q_tail] = ny;
          q_tail++;
        }
      }
    }
  }

  // Find exposed grains (touching air_mask)
  bool exposed[GRID_HEIGHT][GRID_WIDTH] = {0};
  for (int y = 0; y < GRID_HEIGHT; y++) {
    for (int x = 0; x < GRID_WIDTH; x++) {
      if (world[y][x] > 0) {
        if ((y > 0 && air_mask[y - 1][x]) ||
            (y < GRID_HEIGHT - 1 && air_mask[y + 1][x]) ||
            (x > 0 && air_mask[y][x - 1]) ||
            (x < GRID_WIDTH - 1 && air_mask[y][x + 1])) {
          exposed[y][x] = true;
        }
      }
    }
  }

  bool visited[GRID_HEIGHT][GRID_WIDTH] = {0};
  for (int y = 0; y < GRID_HEIGHT; y++) {
    for (int x = 0; x < GRID_WIDTH; x++) {
      uint8_t c = world[y][x];
      if (c > 0 && !visited[y][x]) {
        q_head = 0;
        q_tail = 0;
        visited[y][x] = true;
        q_x[q_tail] = x;
        q_y[q_tail] = y;
        q_tail++;

        int comp_size = 0;
        int min_x = GRID_WIDTH;
        int max_x = -1;

        while (q_head < q_tail) {
          int px = q_x[q_head];
          int py = q_y[q_head];
          q_head++;
          comp_size++;

          if (px < min_x)
            min_x = px;
          if (px > max_x)
            max_x = px;

          int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
          for (int i = 0; i < 4; i++) {
            int nx = px + dirs[i][0];
            int ny = py + dirs[i][1];
            if (nx >= 0 && nx < GRID_WIDTH && ny >= 0 && ny < GRID_HEIGHT) {
              if (!visited[ny][nx] && world[ny][nx] == c) {
                visited[ny][nx] = true;
                q_x[q_tail] = nx;
                q_y[q_tail] = ny;
                q_tail++;
              }
            }
          }
        }

        bool left_touch = (min_x == 0);
        bool right_touch = (max_x == GRID_WIDTH - 1);

        if (left_touch && right_touch) {
          res.cleared += comp_size;
          res.color_gaps[c] = 0;
          continue;
        }

        int target = -1; // 0: LEFT, 1: RIGHT
        if (left_touch) {
          target = 1;
          if ((GRID_WIDTH - 1 - max_x) < res.color_gaps[c])
            res.color_gaps[c] = (GRID_WIDTH - 1 - max_x);
        } else if (right_touch) {
          target = 0;
          if (min_x < res.color_gaps[c])
            res.color_gaps[c] = min_x;
        } else {
          int dL = min_x;
          int dR = GRID_WIDTH - 1 - max_x;
          target = (dL < dR) ? 0 : 1;
          if (target == 0 && dL < res.color_gaps[c])
            res.color_gaps[c] = dL + dR;
          if (target == 1 && dR < res.color_gaps[c])
            res.color_gaps[c] = dL + dR;
        }

        int span = max_x - min_x + 1;
        res.bridge +=
            (left_touch || right_touch) ? (span * 2.0f) : (span * 1.0f);

        int tf_count = 0;
        // iterate component again to find target facing exposed grains
        // wait, we can just do it in the first loop! But we don't know the
        // target until min/max x are found. so we re-iterate over the component
        // by using the queue array.
        for (int i = 0; i < q_tail; i++) {
          int px = q_x[i];
          int py = q_y[i];
          if (exposed[py][px]) {
            if (target == 1) { // RIGHT
              if (px == GRID_WIDTH - 1 || air_mask[py][px + 1])
                tf_count++;
            } else { // LEFT
              if (px == 0 || air_mask[py][px - 1])
                tf_count++;
            }
          }
        }

        res.useful_frontier[c] += tf_count;

        float reach = span / (float)GRID_WIDTH;
        res.flow += tf_count * (reach * reach) * (comp_size / 1000.0f);
      }
    }
  }

  for (int i = 1; i < 5; i++) {
    if (res.color_gaps[i] == GRID_WIDTH) { // empty color
      res.gap += GRID_WIDTH;
    } else {
      res.gap += res.color_gaps[i];
    }
  }

  return res;
}

__attribute__((visibility("default"))) EvalFrontierResult
simulate_action_frontier(uint8_t original_world[GRID_HEIGHT][GRID_WIDTH],
                         int *grains_x, int *grains_y, int num_grains,
                         int color_idx) {
  uint8_t world[GRID_HEIGHT][GRID_WIDTH];
  memcpy(world, original_world, sizeof(world));

  for (int i = 0; i < num_grains; i++) {
    int x = grains_x[i];
    int y = grains_y[i];
    if (x < 0 || x >= GRID_WIDTH || y < 0 || y >= GRID_HEIGHT ||
        world[y][x] != 0) {
      EvalFrontierResult res;
      memset(&res, 0, sizeof(res));
      res.valid = false;
      return res;
    }
  }

  bool can_drop = true;
  int dy = 0;
  while (can_drop) {
    for (int i = 0; i < num_grains; i++) {
      int nx = grains_x[i];
      int ny = grains_y[i] + dy + 1;
      if (ny >= GRID_HEIGHT || world[ny][nx] != 0) {
        can_drop = false;
        break;
      }
    }
    if (can_drop)
      dy++;
  }

  for (int i = 0; i < num_grains; i++) {
    int nx = grains_x[i];
    int ny = grains_y[i] + dy;
    world[ny][nx] = color_idx;
  }

  step_sand(world);
  EvalFrontierResult res = evaluate_frontier_board(world);
  res.valid = true;
  return res;
}

__attribute__((visibility("default"))) void
simulate_all_actions_frontier(uint8_t original_world[GRID_HEIGHT][GRID_WIDTH],
                              int num_actions, int *grains_x_flat,
                              int *grains_y_flat, int *num_grains_array,
                              int color_idx, EvalFrontierResult *out_results) {
  int grain_idx = 0;
  for (int a = 0; a < num_actions; a++) {
    uint8_t world[GRID_HEIGHT][GRID_WIDTH];
    memcpy(world, original_world, sizeof(world));

    int num_grains = num_grains_array[a];
    int *grains_x = &grains_x_flat[grain_idx];
    int *grains_y = &grains_y_flat[grain_idx];
    grain_idx += num_grains;

    bool valid = true;
    for (int i = 0; i < num_grains; i++) {
      int nx = grains_x[i];
      int ny = grains_y[i];
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT ||
          world[ny][nx] != 0) {
        valid = false;
        break;
      }
    }

    if (!valid) {
      memset(&out_results[a], 0, sizeof(EvalFrontierResult));
      out_results[a].valid = false;
      continue;
    }

    bool can_drop = true;
    int dy = 0;
    while (can_drop) {
      for (int i = 0; i < num_grains; i++) {
        int nx = grains_x[i];
        int ny = grains_y[i] + dy + 1;
        if (ny >= GRID_HEIGHT || world[ny][nx] != 0) {
          can_drop = false;
          break;
        }
      }
      if (can_drop)
        dy++;
    }

    for (int i = 0; i < num_grains; i++) {
      int nx = grains_x[i];
      int ny = grains_y[i] + dy;
      world[ny][nx] = color_idx;
    }

    step_sand(world);
    EvalFrontierResult res = evaluate_frontier_board(world);
    res.valid = true;

    // Final game over check if piece is still at top
    bool overflow = false;
    for (int i = 0; i < num_grains; i++) {
      if (grains_y[i] + dy == 0) {
        overflow = true;
        break;
      }
    }
    if (overflow)
      res.valid = false;

    out_results[a] = res;
  }
}

__attribute__((visibility("default"))) void
c_step_sand(uint8_t world[GRID_HEIGHT][GRID_WIDTH]) {
  step_sand(world);
}

__attribute__((visibility("default"))) void
simulate_all_actions_combined(uint8_t original_world[GRID_HEIGHT][GRID_WIDTH],
                              int num_actions, int *grains_x_flat,
                              int *grains_y_flat, int *num_grains_array,
                              int color_idx, EvalResult *out_res_base,
                              EvalFrontierResult *out_res_front) {
  int grain_idx = 0;
  for (int a = 0; a < num_actions; a++) {
    uint8_t world[GRID_HEIGHT][GRID_WIDTH];
    memcpy(world, original_world, sizeof(world));

    int num_grains = num_grains_array[a];
    int grains_x[num_grains];
    int grains_y[num_grains];
    for (int i = 0; i < num_grains; i++) {
      grains_x[i] = grains_x_flat[grain_idx];
      grains_y[i] = grains_y_flat[grain_idx];
      grain_idx++;
    }

    bool valid = true;
    for (int i = 0; i < num_grains; i++) {
      int nx = grains_x[i];
      int ny = grains_y[i];
      if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT ||
          world[ny][nx] != 0) {
        valid = false;
        break;
      }
    }

    if (!valid) {
      memset(&out_res_base[a], 0, sizeof(EvalResult));
      out_res_base[a].valid = false;
      memset(&out_res_front[a], 0, sizeof(EvalFrontierResult));
      out_res_front[a].valid = false;
      continue;
    }

    bool can_drop = true;
    int dy = 0;
    while (can_drop) {
      for (int i = 0; i < num_grains; i++) {
        int nx = grains_x[i];
        int ny = grains_y[i] + dy + 1;
        if (ny >= GRID_HEIGHT || world[ny][nx] != 0) {
          can_drop = false;
          break;
        }
      }
      if (can_drop)
        dy++;
    }

    for (int i = 0; i < num_grains; i++) {
      int nx = grains_x[i];
      int ny = grains_y[i] + dy;
      world[ny][nx] = color_idx;
    }

    step_sand(world);

    EvalResult res_base = evaluate_board(world);
    res_base.valid = true;
    out_res_base[a] = res_base;

    EvalFrontierResult res_front = evaluate_frontier_board(world);
    res_front.valid = true;

    // Final game over check if piece is still at top
    bool overflow = false;
    for (int i = 0; i < num_grains; i++) {
      if (grains_y[i] + dy == 0) {
        overflow = true;
        break;
      }
    }
    if (overflow) {
      res_base.valid = false;
      res_front.valid = false;
      out_res_base[a].valid = false;
      out_res_front[a].valid = false;
    }

    out_res_front[a] = res_front;
  }
}
