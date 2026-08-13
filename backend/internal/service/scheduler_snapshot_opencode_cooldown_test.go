//go:build unit

package service

import (
	"context"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
)

type openCodeCooldownSnapshotRepo struct {
	AccountRepository
	accounts       []Account
	groupID        *int64
	platform       string
	includeGrouped bool
}

func (r *openCodeCooldownSnapshotRepo) ListSchedulerSnapshotCandidates(
	_ context.Context,
	groupID *int64,
	platform string,
	includeGrouped bool,
) ([]Account, error) {
	r.groupID = groupID
	r.platform = platform
	r.includeGrouped = includeGrouped
	return append([]Account(nil), r.accounts...), nil
}

func TestSchedulerSnapshotRetainsOpenCodeGo403CooldownUntilTimestampExpires(t *testing.T) {
	now := time.Now()
	future := now.Add(time.Minute)
	past := now.Add(-time.Minute)
	groupID := int64(11)
	repo := &openCodeCooldownSnapshotRepo{accounts: []Account{
		{
			ID:          1,
			Platform:    PlatformOpenAI,
			Type:        AccountTypeAPIKey,
			Status:      StatusActive,
			Schedulable: true,
			Credentials: map[string]any{
				"quota_limit": float64(1),
				"quota_used":  float64(1),
			},
		},
		{
			ID:                      2,
			Platform:                PlatformOpenAI,
			Type:                    AccountTypeAPIKey,
			Status:                  StatusActive,
			Schedulable:             true,
			TempUnschedulableUntil:  &future,
			TempUnschedulableReason: "OpenAI 403 temporary cooldown (2/3): Access forbidden (403): error code: 1010",
			Credentials: map[string]any{
				"base_url": "https://opencode.ai/zen/go/v1/responses",
			},
		},
		{
			ID:                      3,
			Platform:                PlatformOpenAI,
			Type:                    AccountTypeAPIKey,
			Status:                  StatusActive,
			Schedulable:             true,
			TempUnschedulableUntil:  &future,
			TempUnschedulableReason: "transport error cooldown",
			Credentials: map[string]any{
				"base_url": "https://opencode.ai/zen/go/v1/responses",
			},
		},
		{
			ID:                      4,
			Platform:                PlatformOpenAI,
			Type:                    AccountTypeAPIKey,
			Status:                  StatusActive,
			Schedulable:             true,
			TempUnschedulableUntil:  &past,
			TempUnschedulableReason: "OpenAI 403 temporary cooldown (1/3): expired",
			Credentials: map[string]any{
				"base_url": "https://opencode.ai/zen/go/v1/responses",
			},
		},
	}}
	svc := NewSchedulerSnapshotService(nil, nil, repo, nil, &config.Config{RunMode: config.RunModeStandard})

	accounts, err := svc.loadAccountsFromDB(context.Background(), SchedulerBucket{
		GroupID:  groupID,
		Platform: PlatformOpenAI,
		Mode:     SchedulerModeSingle,
	}, false)

	require.NoError(t, err)
	require.Equal(t, []int64{1, 2, 4}, schedulerSnapshotAccountIDs(accounts))
	require.NotNil(t, repo.groupID)
	require.Equal(t, groupID, *repo.groupID)
	require.Equal(t, PlatformOpenAI, repo.platform)
	require.False(t, repo.includeGrouped)
	require.False(t, accounts[1].IsSchedulable(), "cooldown must still block selection before expiry")
}

func TestSchedulerSnapshotOpenAISimpleModeLoadsAllConfiguredAccounts(t *testing.T) {
	repo := &openCodeCooldownSnapshotRepo{}
	svc := NewSchedulerSnapshotService(nil, nil, repo, nil, &config.Config{RunMode: config.RunModeSimple})

	_, err := svc.loadAccountsFromDB(context.Background(), SchedulerBucket{
		Platform: PlatformOpenAI,
		Mode:     SchedulerModeSingle,
	}, false)

	require.NoError(t, err)
	require.Nil(t, repo.groupID)
	require.True(t, repo.includeGrouped)
}

func schedulerSnapshotAccountIDs(accounts []Account) []int64 {
	ids := make([]int64, 0, len(accounts))
	for i := range accounts {
		ids = append(ids, accounts[i].ID)
	}
	return ids
}
